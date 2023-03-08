# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import csv
import json
import os
import random
import sys

from comet_ml import ExistingExperiment, Experiment

import flowiz as fz
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from flownets import SPyNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from compressai.zoo.image import _load_model
from compressai.entropy_models import EntropyBottleneck
from compressai.models import __CODER_TYPES__, Refinement
from dataloader import VideoData, VideoTestData
from trainer import Trainer
from util.alignment import Alignment
from util.estimate_bpp import estimate_bpp
from util.math import lower_bound
from util.psnr import mse2psnr
from util.sampler import Resampler
from util.seed import seed_everything
from util.vision import PlotFlow, PlotHeatMap, save_image

plot_flow = PlotFlow().cuda()
plot_bitalloc = PlotHeatMap("RB").cuda()

lmda = {1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130, 
        5: 0.0250, 6: 0.0483, 7: 0.0932, 8: 0.1800}

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

class CompressesModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.loss())

        return torch.stack(aux_loss).mean()

class Pframe(CompressesModel):
    def __init__(self, args, mo_coder, res_coder, train_cfg, logger):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') 

        self.if_model = _load_model("mbt2018", "mse", args.quality_level, pretrained=True)
        self.if_model.eval()

        self.MENet = SPyNet(trainable=False)
        self.Motion = mo_coder
        self.Resampler = Resampler()
        self.MCNet = Refinement(8, 64, out_channels=3)

        self.Residual = res_coder

        self.train_cfg = train_cfg

        self.logger = logger

        self.module_dict = {'I'          : self.if_model, 
                            'ME'         : self.MENet, 
                            'Motion'     : self.Motion, 
                            'MC'         : self.MCNet, 
                            'Residual'   : self.Residual
                            }

    def freeze(self, modules):
        '''
            modules (list): contain modules that need to freeze 
        '''
        self.requires_grad_(True)
        for module in modules:
            for param in self.module_dict[module].parameters(): 
                    self.optimizer.state_dict()[param] = {} # remove all state (step, exp_avg, exp_avg_sg)

            self.module_dict[module].requires_grad_(False)
                
    def motion_forward(self, ref_frame, coding_frame):
        flow = self.MENet(ref_frame, coding_frame)
        flow_hat, likelihood_m = self.Motion(flow)
        warped_frame = self.Resampler(ref_frame, flow_hat)
        mc_frame = self.MCNet(flow_hat, ref_frame, warped_frame)

        m_info = {'likelihood_m': likelihood_m, 
                  'flow': flow, 'flow_hat': flow_hat,
                  'warped_frame': warped_frame, 'mc_frame': mc_frame}

        return mc_frame, likelihood_m, m_info

    def forward(self, ref_frame, coding_frame):
        mc_frame, likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame)
        
        r = coding_frame - mc_frame
        r_hat, likelihood_r = self.Residual(r)
        rec_frame = mc_frame + r_hat

        likelihoods = likelihood_m + likelihood_r
        
        return rec_frame, likelihoods, m_info

    def train_2frames(self, batch, loss_on, mode, random_train=False):    
        assert mode in ['motion', 'residual']

        idx = [0, 1]

        if random_train:
            idx[0] = random.randint(0, 6)
            if idx[0] == 0:
                direction = 1
            elif idx[0] == 6:
                direction = -1
            else:
                direction = random.choice([-1, 1])

            idx[1] = idx[0]+direction

        loss = torch.tensor(0., dtype=torch.float, device=batch.device)

        with torch.no_grad():
            out_dir = self.if_model(batch[:, idx[0]])
            ref_frame = out_dir['x_hat']
        
        coding_frame = batch[:, idx[1]]

        if mode == 'motion':
            mc_frame, likelihood_m, m_info = self.motion_forward(ref_frame, coding_frame)

            m_rate = estimate_bpp(likelihood_m, input=coding_frame).mean()

            result = {
                'train/m_rate'              : m_rate,
                'train/warped_distortion'   : self.criterion(coding_frame, m_info['warped_frame']).mean(),
                'train/mc_distortion'       : self.criterion(coding_frame, mc_frame).mean(),
            }
        else:
            rec_frame, likelihoods, m_info = self(ref_frame, coding_frame)
            
            m_rate = estimate_bpp(likelihoods[0], input=coding_frame).mean()
            r_rate = estimate_bpp(likelihoods[1:], input=coding_frame).mean()

            result = {
                'train/m_rate'              : m_rate,
                'train/r_rate'              : r_rate,
                'train/rate'                : m_rate + r_rate,
                'train/warped_distortion'   : self.criterion(coding_frame, m_info['warped_frame']).mean(),
                'train/mc_distortion'       : self.criterion(coding_frame, m_info['mc_frame']).mean(),
                'train/distortion'          : self.criterion(coding_frame, rec_frame).mean(),
            }

        rate = torch.tensor(0., dtype=torch.float, device=batch.device)
        for term in loss_on['R'].split('/'):
            coefficient = 1
            if '*' in term:
                coefficient = float(term.split('*')[0])
                term = term.split('*')[1]

            if term == 'None':
                continue
            
            rate += coefficient * result['train/'+term]

        distortion = torch.tensor(0., dtype=torch.float, device=batch.device)
        for term in loss_on['D'].split('/'):
            coefficient = 1
            if '*' in term:
                coefficient = float(term.split('*')[0])
                term = term.split('*')[1]

            if term == 'None':
                continue
            
            distortion += coefficient * result['train/'+term]

        loss = rate + 255**2 * lmda[self.args.quality_level] * distortion

        result.update({'train/loss': loss})

        return loss, result

    def train_fullgop(self, batch, loss_on, max_num_Pframe=5):   
        total_loss = torch.tensor(0., dtype=torch.float, device=batch.device)
        log = {}

        with torch.no_grad():
            out_dir = self.if_model(batch[:, 0])
            rec_frame = out_dir['x_hat']

        for i in range(1, max_num_Pframe+1):
            coding_frame = batch[:, i]
            ref_frame = rec_frame.detach()

            rec_frame, likelihoods, m_info = self(ref_frame, coding_frame)
            
            m_rate = estimate_bpp(likelihoods[0], input=coding_frame).mean()
            r_rate = estimate_bpp(likelihoods[1:], input=coding_frame).mean()

            result = {
                'train/m_rate'              : m_rate,
                'train/r_rate'              : r_rate,
                'train/rate'                : m_rate + r_rate,
                'train/warped_distortion'   : self.criterion(coding_frame, m_info['warped_frame']).mean(),
                'train/mc_distortion'       : self.criterion(coding_frame, m_info['mc_frame']).mean(),
                'train/distortion'          : self.criterion(coding_frame, rec_frame).mean(),
            }

            rate = torch.tensor(0., dtype=torch.float, device=batch.device)
            for term in loss_on['R'].split('/'):
                coefficient = 1
                if '*' in term:
                    coefficient = float(term.split('*')[0])
                    term = term.split('*')[1]

                if term == 'None':
                    continue
                
                rate += coefficient * result['train/'+term]

            distortion = torch.tensor(0., dtype=torch.float, device=batch.device)
            for term in loss_on['D'].split('/'):
                coefficient = 1
                if '*' in term:
                    coefficient = float(term.split('*')[0])
                    term = term.split('*')[1]

                if term == 'None':
                    continue
                
                distortion += coefficient * result['train/'+term]

            loss = rate + 255**2 * lmda[self.args.quality_level] * distortion
            total_loss += loss

            result.update({'train/loss': loss})
            for k, v in result.items():
                if k not in log:
                    log[k] = []
                
                log[k].append(v)
            
        for k in log.keys():
            log[k] = sum(log[k]) / len(log[k])

        total_loss /= max_num_Pframe
        
        return total_loss, log

    def training_step(self, batch, phase):
        device = next(self.parameters()).device
        batch = batch.to(device)

        self.freeze(self.train_cfg[phase]['frozen_modules'])

        if self.train_cfg[phase]['strategy']['stage'] == '2frames':            
            loss, result = self.train_2frames(batch, 
                                              self.train_cfg[phase]['loss_on'], 
                                              self.train_cfg[phase]['mode'],
                                              bool(self.train_cfg[phase]['strategy']['random']))

        elif self.train_cfg[phase]['strategy']['stage'] == 'fullgop':
            loss, result = self.train_fullgop(batch,
                                              self.train_cfg[phase]['loss_on'],
                                              self.train_cfg[phase]['strategy']['max_num_Pframe'])
        else:
            raise NotImplementedError

        return loss, result

    def training_step_end(self, result, epoch=None):
        logs = {}
        for k, v in result.items():
            if k[-10:] == 'distortion':
                logs[k.replace('distortion', 'PSNR')] = mse2psnr(v.item())
            else:
                logs[k] = v.item()
        
        self.logger.log_metrics(logs, epoch=epoch)

    @torch.no_grad()
    def validation_step(self, batch, epoch):
        def create_grid(img):
            return make_grid(torch.unsqueeze(img, 1)).cpu().detach().numpy()[0]

        def upload_img(tnsr, tnsr_name, current_epoch, ch="first", grid=True):
            if grid:
                tnsr = create_grid(tnsr)

            self.logger.log_image(tnsr, name=tnsr_name, step=current_epoch,
                                  image_channels=ch, overwrite=True)

        dataset_name, seq_name, batch, frame_id_start = batch
        device = next(self.parameters()).device
        batch = batch.to(device)

        seq_name = seq_name
        dataset_name = dataset_name

        gop_size = batch.size(1)

        m_rate_list = [[] for _ in range(len(dataset_name))]
        rate_list = [[] for _ in range(len(dataset_name))]
        psnr_list = [[] for _ in range(len(dataset_name))]
        mc_psnr_list = [[] for _ in range(len(dataset_name))]
        mse_list = [[] for _ in range(len(dataset_name))]
        loss_list = [[] for _ in range(len(dataset_name))]

        align = Alignment(64)

        for frame_idx in range(gop_size):
            coding_frame = batch[:, frame_idx]

            # I frame
            if frame_idx == 0:
                out_dir = self.if_model(align.align(coding_frame))
                rec_frame, likelihoods = out_dir['x_hat'], (out_dir['likelihoods']['y'], out_dir['likelihoods']['z'])
                rec_frame = align.resume(rec_frame).clamp(0, 1)

                for i in range(len(dataset_name)):
                    r_y = estimate_bpp(likelihoods[0][i:i+1], input=coding_frame).mean().cpu().item()
                    r_z = estimate_bpp(likelihoods[1][i:i+1], input=coding_frame).mean().cpu().item()
                    rate = r_y + r_z

                    mse = self.criterion(rec_frame[i], coding_frame[i]).mean().cpu().item()
                    psnr = mse2psnr(mse)
                    loss = rate + 255**2 * lmda[self.args.quality_level] * mse

                    rate_list[i].append(rate)
                    psnr_list[i].append(psnr)
                    mse_list[i].append(mse)
                    loss_list[i].append(loss)
            
            # P frame
            else:
                rec_frame, likelihoods, m_info = self(align.align(ref_frame), align.align(coding_frame))
                rec_frame = align.resume(rec_frame).clamp(0, 1)

                mc_frame = align.resume(m_info['mc_frame']).clamp(0, 1)

                for i, seq in enumerate(seq_name):
                    m_y = estimate_bpp(likelihoods[0][i:i+1], input=coding_frame).mean().cpu().item()
                    r_y = estimate_bpp(likelihoods[1][i:i+1], input=coding_frame).mean().cpu().item()
                    r_z = estimate_bpp(likelihoods[2][i:i+1], input=coding_frame).mean().cpu().item()

                    m_rate = m_y
                    r_rate = r_y + r_z
                    rate = m_rate + r_rate

                    mse = self.criterion(rec_frame[i], coding_frame[i]).mean().item()
                    psnr = mse2psnr(mse)
                    mc_mse = self.criterion(mc_frame[i], coding_frame[i]).mean().item()
                    mc_psnr = mse2psnr(mc_mse)
                    loss = rate + 255**2 * lmda[self.args.quality_level] * mse 

                    rate_list[i].append(rate)
                    psnr_list[i].append(psnr)
                    mse_list[i].append(mse)
                    loss_list[i].append(loss)
                    m_rate_list[i].append(m_rate)
                    mc_psnr_list[i].append(mc_psnr)

                    if frame_idx == 5:
                        if seq not in ['Beauty', 'Jockey', 'Kimono1', 'HoneyBee']:
                            continue

                        flow = align.resume(m_info['flow'])[i]
                        flow_rgb = torch.from_numpy(
                            fz.convert_from_flow(flow.permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                        upload_img(flow_rgb.cpu().numpy(), f'{seq}_{epoch}_est_flow.png', current_epoch=epoch, grid=False)

                        flow_hat = align.resume(m_info['flow_hat'])[i]
                        flow_rgb = torch.from_numpy(
                            fz.convert_from_flow(flow_hat.permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                        upload_img(flow_rgb.cpu().numpy(), f'{seq}_{epoch}_flow_hat.png', current_epoch=epoch, grid=False)

                        warped_mse = self.criterion(align.resume(m_info['warped_frame'])[i], coding_frame[i]).mean().item()
                        warped_psnr = mse2psnr(warped_mse)
                        upload_img(align.resume(m_info['warped_frame'])[i].cpu().numpy(), f'{seq}_{epoch}_{warped_psnr:.3f}_warped_frame.png', current_epoch=epoch, grid=False)
                        upload_img(align.resume(ref_frame)[i].cpu().numpy(), f'{seq}_{epoch}_ref_frame.png', current_epoch=epoch, grid=False)
                        upload_img(align.resume(coding_frame)[i].cpu().numpy(), f'{seq}_{epoch}_gt_frame.png', current_epoch=epoch, grid=False)
                        upload_img(align.resume(rec_frame)[i].cpu().numpy(), f'{seq}_{epoch}_{psnr:.3f}_rec_frame.png', current_epoch=epoch, grid=False)
                        upload_img(align.resume(mc_frame)[i].cpu().numpy(), f'{seq}_{epoch}_{mc_psnr:.3f}_mc_frame.png', current_epoch=epoch, grid=False)

                        # upload rate map
                        cm = plt.get_cmap('hot')
                        lll = lower_bound(likelihoods[0][i:i+1], 1e-9).log() / -np.log(2.)
                        rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                        upload_img(np.transpose(rate_map[:, :, :3], (2, 0, 1)), f'{seq}_{epoch}_my.png', current_epoch=epoch, grid=False)

                        lll = lower_bound(likelihoods[1][i:i+1], 1e-9).log() / -np.log(2.)
                        rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                        upload_img(np.transpose(rate_map[:, :, :3], (2, 0, 1)), f'{seq}_{epoch}_ry.png', current_epoch=epoch, grid=False)

                        lll = lower_bound(likelihoods[2][i:i+1], 1e-9).log() / -np.log(2.)
                        rate_map = cm(lll.cpu().numpy().mean(axis=1)[0])
                        upload_img(np.transpose(rate_map[:, :, :3], (2, 0, 1)), f'{seq}_{epoch}_rz.png', current_epoch=epoch, grid=False)

            ref_frame = rec_frame

        m_rate  = [np.mean(l) for l in m_rate_list]
        rate    = [np.mean(l) for l in rate_list]
        psnr    = [np.mean(l) for l in psnr_list]
        mc_psnr = [np.mean(l) for l in mc_psnr_list]
        mse     = [np.mean(l) for l in mse_list]
        loss    = [np.mean(l) for l in loss_list]

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 
                'val/loss': loss, 'val/mse': mse, 
                'val/psnr': psnr, 'val/mc_psnr': mc_psnr,
                'val/rate': rate, 'val/m_rate': m_rate}

        return logs

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        rd_dict = {}
        loss = []

        for logs in outputs:
            for i in range(len(logs['dataset_name'])):
                dataset_name = logs['dataset_name'][i]

                if not (dataset_name in rd_dict.keys()):
                    rd_dict[dataset_name] = {}
                    rd_dict[dataset_name]['psnr'] = []
                    rd_dict[dataset_name]['rate'] = []
                    rd_dict[dataset_name]['mc_psnr'] = []
                    rd_dict[dataset_name]['m_rate'] = []

                rd_dict[dataset_name]['psnr'].append(logs['val/psnr'][i])
                rd_dict[dataset_name]['rate'].append(logs['val/rate'][i])
                rd_dict[dataset_name]['mc_psnr'].append(logs['val/mc_psnr'][i])
                rd_dict[dataset_name]['m_rate'].append(logs['val/m_rate'][i])
    
                loss.append(logs['val/loss'][i])

        avg_loss = np.mean(loss)
        
        logs = {'val/loss': avg_loss}

        for dataset_name, rd in rd_dict.items():
            logs['val/'+dataset_name+' psnr'] = np.mean(rd['psnr'])
            logs['val/'+dataset_name+' rate'] = np.mean(rd['rate'])
            logs['val/'+dataset_name+' mc_psnr'] = np.mean(rd['mc_psnr'])
            logs['val/'+dataset_name+' m_rate'] = np.mean(rd['m_rate'])

        self.logger.log_metrics(logs)

    @torch.no_grad()
    def test_step(self, batch):
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate', 'MC-PSNR', 'Warped-PSNR']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        log_list = []
        align = Alignment(64)

        dataset_name, seq_name, batch, frame_id_start = batch
        device = next(self.parameters()).device
        batch = batch.to(device)

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        os.makedirs(self.args.save_dir + f'/{seq_name}/flow_hat', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/warped_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)

        for frame_idx in range(gop_size):
            TO_VISUALIZE = False and frame_id_start == 1 and frame_idx < 8 and seq_name in ['BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey']
            
            coding_frame = batch[:, frame_idx]

            # I frame
            if frame_idx == 0:
                out_dir = self.if_model(align.align(coding_frame))
                rec_frame, likelihoods = out_dir['x_hat'], (out_dir['likelihoods']['y'], out_dir['likelihoods']['z'])
                rec_frame = align.resume(rec_frame).clamp(0, 1)

                r_y = estimate_bpp(likelihoods[0], input=coding_frame).mean().item()
                r_z = estimate_bpp(likelihoods[1], input=coding_frame).mean().item()
                rate = r_y + r_z

                mse = self.criterion(rec_frame, coding_frame).mean().item()
                psnr = mse2psnr(mse)
                log_list.append({'PSNR': psnr, 'Rate': rate})
            
            # P frame
            else:
                rec_frame, likelihoods, m_info = self(align.align(ref_frame), align.align(coding_frame))
                rec_frame = align.resume(rec_frame).clamp(0, 1)

                mc_frame = align.resume(m_info['mc_frame']).clamp(0, 1)
                warped_frame = align.resume(m_info['warped_frame']).clamp(0, 1)
                
                m_y = estimate_bpp(likelihoods[0], input=coding_frame).cpu().item()
                r_y = estimate_bpp(likelihoods[1], input=coding_frame).cpu().item()
                r_z = estimate_bpp(likelihoods[2], input=coding_frame).cpu().item()

                m_rate = m_y
                r_rate = r_y + r_z
                rate = m_rate + r_rate

                mse = self.criterion(rec_frame, coding_frame).mean().item()
                psnr = mse2psnr(mse)
                mc_mse = self.criterion(mc_frame, coding_frame).mean().item()
                mc_psnr = mse2psnr(mc_mse)
                warped_mse = self.criterion(warped_frame, coding_frame).mean().item()
                warped_psnr = mse2psnr(warped_mse)

                if TO_VISUALIZE:
                    flow_map = plot_flow(align.resume(m_info['flow_hat']))
                    save_image(flow_map,
                               self.args.save_dir + f'/{seq_name}/flow_hat/'
                                                    f'frame_{int(frame_id_start + frame_idx)}_flow.png', nrow=1)
                    save_image(coding_frame[0], 
                               self.args.save_dir + f'/{seq_name}/gt_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(mc_frame[0], 
                               self.args.save_dir + f'/{seq_name}/mc_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(warped_frame[0], 
                               self.args.save_dir + f'/{seq_name}/warped_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(ref_frame[0], 
                               self.args.save_dir + f'/{seq_name}/rec_frame/'
                                                    f'frame_{int(frame_id_start + frame_idx)}.png')


                log_list.append({'PSNR': psnr, 'Rate': rate, 'MC-PSNR': mc_psnr, 'Warped-PSNR': warped_psnr,
                                 'my': estimate_bpp(likelihoods[0], input=ref_frame).cpu().item(), 
                                 'ry': estimate_bpp(likelihoods[1], input=ref_frame).cpu().item(), 
                                 'rz': estimate_bpp(likelihoods[2], input=ref_frame).cpu().item()})

                metrics['Mo_Rate'].append(m_rate)
                metrics['MC-PSNR'].append(mc_psnr)
                metrics['Warped-PSNR'].append(warped_psnr)

            ref_frame = rec_frame

            metrics['PSNR'].append(psnr)
            metrics['Rate'].append(rate)

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list}
        return logs

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        metrics_name = list(outputs[0]['metrics'].keys())  # Get all metrics' names

        rd_dict = {}

        single_seq_logs = {}
        for metrics in metrics_name:
            single_seq_logs[metrics] = {}

        single_seq_logs['LOG'] = {}
        single_seq_logs['GOP'] = {}  # Will not be printed currently
        single_seq_logs['Seq_Names'] = []

        for logs in outputs:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                
                for metrics in metrics_name:
                    rd_dict[dataset_name][metrics] = []

            for metrics in logs['metrics'].keys():
                rd_dict[dataset_name][metrics].append(logs['metrics'][metrics])

            # Initialize
            if seq_name not in single_seq_logs['Seq_Names']:
                single_seq_logs['Seq_Names'].append(seq_name)
                for metrics in metrics_name:
                    single_seq_logs[metrics][seq_name] = []
                single_seq_logs['LOG'][seq_name] = []
                single_seq_logs['GOP'][seq_name] = []

            # Collect metrics logs
            for metrics in metrics_name:
                single_seq_logs[metrics][seq_name].append(logs['metrics'][metrics])
            single_seq_logs['LOG'][seq_name].extend(logs['log_list'])
            single_seq_logs['GOP'][seq_name] = len(logs['log_list'])

        os.makedirs(self.args.save_dir + f'/report', exist_ok=True)

        for seq_name, log_list in single_seq_logs['LOG'].items():
            with open(self.args.save_dir + f'/report/{seq_name}.csv', 'w', newline='') as report:
                writer = csv.writer(report, delimiter=',')
                columns = ['frame'] + list(log_list[1].keys())
                writer.writerow(columns)

                for idx in range(len(log_list)):
                    writer.writerow([f'frame_{idx + 1}'] + list(log_list[idx].values()))

        # Summary
        logs = {}
        print_log = '{:>16} '.format('Sequence_Name')
        for metrics in metrics_name:
            print_log += '{:>12}'.format(metrics)
        print_log += '\n'

        for seq_name in single_seq_logs['Seq_Names']:
            print_log += '{:>16} '.format(seq_name[:5])

            for metrics in metrics_name:
                print_log += '{:12.4f}'.format(np.mean(single_seq_logs[metrics][seq_name]))

            print_log += '\n'
        print_log += '================================================\n'
        for dataset_name, rd in rd_dict.items():
            print_log += '{:>16} '.format(dataset_name)

            for metrics in metrics_name:
                logs['test/' + dataset_name + ' ' + metrics] = np.mean(rd[metrics])
                print_log += '{:12.4f}'.format(np.mean(rd[metrics]))

            print_log += '\n'

        print(print_log)

        with open(self.args.save_dir + f'/brief_summary.txt', 'w', newline='') as report:
            report.write(print_log)

        self.logger.log_metrics(logs)

    def optimizer_step(self):
        def clip_gradient(opt, grad_clip):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(self.optimizer, 5)

        self.optimizer.step()

    def configure_optimizers(self, lr):
        parameters = {n for n, _ in self.named_main_parameters()}
        aux_parameters = {n for n, _ in self.named_aux_parameters()}

        # Make sure we don't have an intersection of parameters
        params_dict = dict(self.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        self.optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=lr)
        self.aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=10*lr)
    
    def setup(self, stage):
        dataset_root = os.getenv('DATAROOT')

        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VideoData(dataset_root + "/vimeo_septuplet/", 7, transform=transformer) 
            self.val_dataset = VideoTestData(dataset_root, sequence=('U', 'B'), first_gop=True, GOP=self.args.gop)
        
        elif stage == 'test':
            self.test_dataset = VideoTestData(dataset_root, sequence=('U', 'B'), GOP=self.args.gop)

        else:
            raise NotImplementedError

    def train_dataloader(self, batch_size):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self, batch_size):
        val_loader = DataLoader(self.val_dataset,
                                batch_size=batch_size,
                                num_workers=self.args.num_workers,
                                drop_last=True,
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        return test_loader

    def parallel(self, device_ids):
        self.if_model   = CustomDataParallel(self.if_model, device_ids=device_ids)
        self.MENet      = CustomDataParallel(self.MENet, device_ids=device_ids)
        self.Motion     = CustomDataParallel(self.Motion, device_ids=device_ids)
        self.MCNet      = CustomDataParallel(self.MCNet, device_ids=device_ids)
        self.Residual   = CustomDataParallel(self.Residual, device_ids=device_ids)
        

def parse_args(argv):
    parser = argparse.ArgumentParser()

    # training specific
    parser.add_argument('--MENet', type=str, choices=['PWC', 'SPy'], default='SPy')
    parser.add_argument('--train_conf', type=str, default=None)
    parser.add_argument('--motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("-q", "--quality_level", type=int, default=6, help="Quality level (default: %(default)s)")

    parser.add_argument("--gpus", type=int, default=1, help="Number of GPU (default: %(default)s)")
    parser.add_argument('--save_dir', default=None, help='directory for saving testing result')
    parser.add_argument('--gop', default=12, type=int)

    parser.add_argument('--project_name', type=str, default='Video_Compression')
    parser.add_argument('--experiment_name', type=str, default='DVC')
    parser.add_argument('--restore', type=str, default='none', choices=['none', 'load'])
    parser.add_argument('--restore_exp_key', type=str, default=None)
    parser.add_argument('--restore_exp_epoch', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_sanity', action='store_true')

    args = parser.parse_args(argv)

    return args

def main(argv):
    args = parse_args(argv)
    assert args.gpus <= torch.cuda.device_count(), "Can't find enough gpus in the machine."

    save_root = os.getenv('LOG', './') 

    if args.save_dir is None:
        args.save_dir = os.path.join(save_root, args.project_name, args.experiment_name + '-' + str(args.quality_level))

    exp = Experiment 
    experiment = exp(
        api_key="sriOLxa6VvcxCPgGaKaaxAk0p",
        project_name=args.project_name,
        workspace="hongsheng416",
        experiment_key = None,
        disabled=args.debug or args.test
    )
    experiment.set_name(f'{args.experiment_name}-{args.quality_level}')
    experiment.log_parameters(vars(args))
    ckpt_dir = os.path.join(save_root, args.project_name, experiment.get_key(), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(888888)
    
    gpu_ids = [0]
    for i in range(1, args.gpus):
        gpu_ids.append(i)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # log model config
    for cfg in [args.motion_coder_conf, args.residual_coder_conf]:
        if os.path.exists(cfg):
            experiment.log_code(cfg)

    # train config
    if args.train_conf is not None:
        experiment.log_code(args.train_conf)
        with open(args.train_conf, 'r') as jsonfile:
            train_cfg = json.load(jsonfile)
    else:
        train_cfg = None

    # Config coders
    assert not (args.motion_coder_conf is None)
    mo_coder_cfg = yaml.safe_load(open(args.motion_coder_conf, 'r'))
    assert mo_coder_cfg['model_architecture'] in __CODER_TYPES__.keys()
    mo_coder_arch = __CODER_TYPES__[mo_coder_cfg['model_architecture']]
    mo_coder = mo_coder_arch(**mo_coder_cfg['model_params'])

    assert not (args.residual_coder_conf is None)
    res_coder_cfg = yaml.safe_load(open(args.residual_coder_conf, 'r'))
    assert res_coder_cfg['model_architecture'] in __CODER_TYPES__.keys()
    res_coder_arch = __CODER_TYPES__[res_coder_cfg['model_architecture']]
    res_coder = res_coder_arch(**res_coder_cfg['model_params'])
    
    model = Pframe(args, mo_coder, res_coder, train_cfg, experiment).to(device)

    if args.restore == 'load':
        checkpoint = torch.load(os.path.join(save_root, args.project_name, args.restore_exp_key, 'checkpoints', f'epoch={args.restore_exp_epoch}.pth.tar'), map_location=device)
        current_epoch = 1

        ckpt = {}
        for k, v in checkpoint["state_dict"].items():
            k = k.split('.')
            k.pop(1)
            if k[1] == 'entropy_bottleneck'and k[2] in ['_offset', '_quantized_cdf', '_cdf_length']:
                    continue
            ckpt['.'.join(k)] = v    

        model.load_state_dict(ckpt, strict=False)
    else: 
        current_epoch = 1

    if args.gpus >= 1 and torch.cuda.device_count() >= 1:
        model.parallel(device_ids=gpu_ids)

    trainer = Trainer(args, model, train_cfg, current_epoch, ckpt_dir, device)
    
    if args.test:
        trainer.test()
    else:
        if not args.debug:
            try:
                trainer.fit()
                experiment.send_notification(
                    f"Experiment {args.experiment_name}-{args.quality_level} ({experiment.get_key()})",
                    "completed successfully"
                )
            except Exception as e:
                print(e)
                torch.save({
                    "epoch": trainer.current_epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": model.optimizer.state_dict(),
                }, 
                f'{ckpt_dir}/last.pth.tar')

                experiment.send_notification(
                    f"Experiment {args.experiment_name}-{args.quality_level} ({experiment.get_key()})",
                    "failed"
                )
        else:
            trainer.fit()

if __name__ == "__main__":
    main(sys.argv[1:])

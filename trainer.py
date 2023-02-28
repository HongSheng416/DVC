import shutil
import gc
from itertools import count

import numpy as np
import torch

from tqdm import tqdm

class Trainer():
    def __init__(self, args, model, train_cfg, current_epoch, save_root, device):
        super(Trainer, self).__init__()
        self.args =args
        self.model = model
        self.train_cfg = train_cfg

        if train_cfg is not None:
            self.phase = {}
            for k, v in sorted({v: k for k, v in train_cfg['phase'].items()}.items()):
                self.phase[k] = v

        self.current_epoch = current_epoch
        self.current_phase = None
        self.save_root = save_root
        self.num_device = 1 if device == 'cpu' else args.gpus

    def get_phase(self, epoch):
        for k in self.phase.keys():
            if epoch <= k:
                return self.phase[k]
                
    def get_prev_phase(self, epoch):
        previous = None
        current = None
        for k in self.phase.keys():
            previous = current
            current = self.phase[k]

            if epoch <= k:
                return previous

    def save_checkpoint(self, state, is_best):
        torch.save(state, self.save_root + f'/epoch={self.current_epoch}.pth.tar')
        if is_best:
            shutil.copyfile(self.save_root + f'/epoch={self.current_epoch}.pth.tar',
                            self.save_root + f'/checkpoint_best_loss.pth.tar')

    def fit(self):
        self.model.setup('fit')

        if not self.args.no_sanity:
            self.before_train()

        # setup val dataloader
        self.val_loader = self.model.val_dataloader(self.num_device)

        start = self.current_epoch
        best_loss = float("inf")

        for epoch in count(start):            
            phase = self.get_phase(epoch)

            if phase != self.current_phase:
                # setup train dataloader
                self.train_loader = self.model.train_dataloader(self.train_cfg[phase]['batch_size'] * self.num_device)

                # setup optimizer
                lr = self.train_cfg[phase]['lr']
                self.model.configure_optimizers(lr)

                self.current_phase = phase

                print(f'Start {self.current_phase} phase.\n lr: {lr}, frozen_modules: {self.train_cfg[phase]["frozen_modules"]}')

            self.current_epoch = epoch
            self.model.train()

            # setup train progressbar
            progressbar = tqdm(self.train_loader, total=len(self.train_loader))
            progressbar.set_description(f'epoch {epoch}')

            for batch in progressbar:
                self.model.optimizer.zero_grad()
                self.model.aux_optimizer.zero_grad()

                loss, logs = self.model.training_step(batch, phase)
                loss.backward()
                self.model.optimizer_step()

                aux_loss = self.model.aux_loss()
                aux_loss.backward()
                self.model.aux_optimizer.step()

                self.model.training_step_end(logs, epoch)

                update_txt=f'Loss: {loss.item():.3f}, Aux loss: {aux_loss.item():.3f}'
                progressbar.set_postfix_str(update_txt, refresh=True)

                del batch, loss, logs

            self.model.eval()
            outputs = []
            val_loss = []
            
            # setup validation progressbar
            progressbar = tqdm(self.val_loader, total=len(self.val_loader), leave=True)
            progressbar.set_description(f'epoch {epoch}')        

            torch.cuda.empty_cache()  

            for batch in progressbar:
                logs = self.model.validation_step(batch, epoch)
                outputs.append(logs)
                val_loss.append(np.mean(logs['val/loss']))

                update_txt=f'[Validation Loss: {np.mean(logs["val/loss"]):.3f}]'
                progressbar.set_postfix_str(update_txt, refresh=True)

            self.model.validation_epoch_end(outputs)
            torch.cuda.empty_cache()

            val_loss = np.mean(val_loss)
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            self.save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "loss": val_loss,
                    "optimizer": self.model.optimizer.state_dict(),
                },
                is_best
            )
            
    def test(self):
        # setup dataloader
        self.model.setup('test')
        test_loader = self.model.test_dataloader()

        self.model.eval()
        outputs = []
        for batch in tqdm(test_loader):
            logs = self.model.test_step(batch)
            outputs.append(logs)

        self.model.test_epoch_end(outputs)

    def before_train(self):
        # setup val dataloader
        self.val_loader = self.model.val_dataloader(self.num_device)

        self.model.eval()
        outputs = []
        val_loss = []
        
        # setup validation progressbar
        progressbar = tqdm(self.val_loader, total=len(self.val_loader), leave=True)
        progressbar.set_description(f'epoch 0')

        for batch in progressbar:
            logs = self.model.validation_step(batch, 0)
            outputs.append(logs)
            val_loss.append(np.mean(logs["val/loss"]))

            update_txt=f'[Validation Loss: {np.mean(logs["val/loss"]):.3f}]'
            progressbar.set_postfix_str(update_txt, refresh=True)

        torch.cuda.empty_cache()

        self.model.validation_epoch_end(outputs)
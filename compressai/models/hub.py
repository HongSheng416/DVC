import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers.gdn import GDN
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.models.utils import conv, deconv
from .google import FactorizedPrior, ScaleHyperprior

class GoogleAnalysisTransform(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size):
        super(GoogleAnalysisTransform, self).__init__(
            conv(in_channels, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters),
            conv(num_filters, num_features, kernel_size, stride=2)
        )

class GoogleSynthesisTransform(nn.Sequential):
    def __init__(self, out_channels, num_features, num_filters, kernel_size):
        super(GoogleSynthesisTransform, self).__init__(
            deconv(num_features, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            deconv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            deconv(num_filters, num_filters, kernel_size, stride=2),
            GDN(num_filters, inverse=True),
            deconv(num_filters, out_channels, kernel_size, stride=2)
        )

class GoogleHyperAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperAnalysisTransform, self).__init__(
            conv(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(num_filters, num_hyperpriors, kernel_size=5, stride=2)
        )

class GoogleHyperSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransform, self).__init__(
            deconv(num_hyperpriors, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(num_filters, num_features, kernel_size=3, stride=1)
        )

class GoogleFactorizedPrior(FactorizedPrior):
    def __init__(self, in_channels=2, out_channels=2, kernel_size=3,
                 num_filters=128, num_features=128, **kwargs):
        super().__init__(num_filters, num_features, **kwargs)

        self.g_a = GoogleAnalysisTransform(in_channels, num_filters, num_features, kernel_size)
        self.g_s = GoogleSynthesisTransform(out_channels, num_features, num_filters, kernel_size)

    def forward(self, x):
        result = super().forward(x)
        return result['x_hat'], (result['likelihoods']['y'],)

class GoogleHyperPrior(ScaleHyperprior):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, 
                 num_filters=128, num_features=128, num_hyperpriors=128, **kwargs):
        super().__init__(num_filters, num_features, **kwargs)

        self.g_a = GoogleAnalysisTransform(in_channels, num_features, num_filters, kernel_size)
        self.g_s = GoogleSynthesisTransform(out_channels, num_features, num_filters, kernel_size)
        self.h_a = GoogleHyperAnalysisTransform(num_features, num_filters, num_hyperpriors)
        self.h_s = GoogleHyperSynthesisTransform(num_features, num_filters, num_hyperpriors)

        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(num_hyperpriors)

    def forward(self, x):
        result = super().forward(x)
        return result['x_hat'], (result['likelihoods']['y'], result['likelihoods']['z'])

__CODER_TYPES__ = {'GoogleFactorizedPrior': GoogleFactorizedPrior,
                   'GoogleHyperPrior'     : GoogleHyperPrior}
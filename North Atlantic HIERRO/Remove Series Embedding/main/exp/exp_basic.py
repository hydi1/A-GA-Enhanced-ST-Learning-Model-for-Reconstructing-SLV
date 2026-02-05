import os

import torch

from models import GAconvgru

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {

            'GAconvgru': GAconvgru
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if 'use_gpu' in self.args and self.args['use_gpu']:

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:

            device = torch.device('cpu')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

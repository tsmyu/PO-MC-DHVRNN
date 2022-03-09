'''
メモ：
パラメータからネットワーク復元
input->run
params.p state_best_dict.pth -> model.npz obs_normalizer.npz optimizer.npz を作る
'''

import pickle
import torch
import torch.nn as nn
import numpy as np

from vrnn.models.rnn_gauss import RNN_GAUSS
from vrnn.models.macro_vrnn import MACRO_VRNN


def load_model(model_name, params, parser=None):
    model_name = model_name.lower()

    if model_name == 'rnn_gauss':
        return RNN_GAUSS(params, parser)
    elif model_name == 'macro_vrnn':
        return MACRO_VRNN(params, parser)
    else:
        raise NotImplementedError


def reconstruction_model():
    params = pickle.load(open(
        r'C:\Users\cgsc1013\Desktop\IL\PO-MC-DHVRNN\weights\sub60_filt_vel_meanHMM_acc_0_unnorm\RNN_GAUSS_bat\att_-1_10_96\params.p', 'rb'))

    state_dict = torch.load(
        r'C:\Users\cgsc1013\Desktop\IL\PO-MC-DHVRNN\weights\sub60_filt_vel_meanHMM_acc_0_unnorm\RNN_GAUSS_bat\att_-1_10_96_state_dict_best.pth')

    model = load_model(params['model'], params)
    model.load_state_dict(state_dict)

    return state_dict, model


def make_optimizer(state_dict):
    opt = state_dict

    return opt


if __name__ == '__main__':
    state_dict, model = reconstruction_model()
    optimizer = make_optimizer(state_dict)

    np.save(r'C:\Users\cgsc1013\Desktop\IL\PO-MC-DHVRNN\weights\sub60_filt_vel_meanHMM_acc_0_unnorm\RNN_GAUSS_bat\att_-1_10_96_npz\model.npz', model)
    np.save(r'C:\Users\cgsc1013\Desktop\IL\PO-MC-DHVRNN\weights\sub60_filt_vel_meanHMM_acc_0_unnorm\RNN_GAUSS_bat\att_-1_10_96_npz\optimizer.npz', optimizer)

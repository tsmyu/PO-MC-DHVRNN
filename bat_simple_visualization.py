import pickle
from select import select
import numpy as np
import os
import argparse
import torch
import matplotlib.pyplot as plt
from vrnn.models import rnn_gauss, load_model


class ShowData:
    def reconstruction_model(self):
        '''
        モデルを入力する
        '''
        weights_path = r'.\weights\sub60_filt_vel_meanHMM_acc_0_unnorm\RNN_GAUSS_bat'
        state_dict_path = weights_path + r'\att_-1_10_96_state_dict_best.pth'
        params_path = weights_path + r'\att_-1_10_96\params.p'
        self.params = pickle.load(open(params_path, 'rb'))
        state_dict = torch.load(state_dict_path)
        self.model = load_model(self.params['model'], self.params, parser=None)
        self.model.load_state_dict(state_dict)

        return state_dict, self.model, self.params

    def prepare_input_data(self):
        '''
        モデルに入れるデータを準備する
        2022/05/24時点では10次元:位置、速度、水平方向・垂直方向の旋回角度、障害物との距離と角度
        最初はランダム？
        model.sampleつかう入力のstateは最初はinput_state1にする
        出力は今ほしいのはデータそのものだからy_tを使う
        '''
        input_state_path = r'./data/all_bat_games_100_unnorm_filt_acc_k0_meanHMM/Fs60_vel_10_96_tr0.pkl'
        state_0 = pickle.load(open(input_state_path, 'rb'))
        # state_0 = state_0[0][0][:][:][:]
        # print(state_0[0])
        state_0 = np.array(state_0[0])
        input_state = torch.from_numpy(state_0)
        input_state = input_state.permute(2, 1, 0, 3)
        # state, _, _, _, _ = self.model.sample(input_state, rollout=True, burn_in=self.params['burn_in'], L_att=False, CF_pred=False, n_sample=self.params['n_agents'], TEST=True)
        # next_state = state

        return input_state

    # def plot_data(self):


if __name__ == '__main__':
    SD = ShowData()
    _, model, params = SD.reconstruction_model()
    input_state = SD.prepare_input_data()
    # print(params)
    for i in range(1):
        states, _, _, _, _, prediction = model.sample(
            input_state, rollout=True, burn_in=params['burn_in'], L_att=False, CF_pred=True, n_sample=10, TEST=True)
        print(prediction)
    # flag = False

    # if states == input_state:
    #     flag = True

    # print(flag)

    import pdb; pdb.set_trace()
    # print(state.size())
    # print(state)

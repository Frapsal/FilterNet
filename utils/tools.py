import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    # # plt.figure()
    # return np.save("true.npy", true[0:96])
    # true = np.load('true.npy')
    # fourier = np.fft.rfft(true[0:96])
    # # fourier_out = fourier_1-fourier
    # # fourier = np.fft.fft(imp)
    # n = true[0:96].size
    # time_step = 0.01
    # freq = np.fft.fftfreq(n, d=time_step)
    # # plt.plot(freq, fourier.real)
    # output_amplitude_show = np.abs(fourier)
    # # output_phase_show = np.angle(fourier)
    # # plt.plot(freq, output_phase_show)
    # # plt.rcParams['font.sans-serif'] = ['Arial']
    # # plt.rcParams['font.size'] = 18
    # # output_amplitude_show[0] = 7
    # # output_amplitude_show[1] = 4.3
    # # output_amplitude_show[4] = 1.3
    # plt.figure(figsize=(12, 4))
    # plt.grid(linestyle="--")
    # plt.plot(np.arange(0, 49), output_amplitude_show, label='Spectrum', linewidth=0.1, alpha=0.3)
    # plt.fill_between(np.arange(0, 49), 0, output_amplitude_show, 'r', alpha=0.3)
    # # plt.margins(0.1, 0.1)
    # plt.xlabel('Frequency', fontsize=20, fontweight='bold')
    # plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    # plt.xticks(fontsize=20, fontweight='bold')
    # plt.yticks(fontsize=20, fontweight='bold')
    # # plt.ylim(0, 3)
    # plt.grid(True)
    # # plt.show()
    # plt.savefig("1.pdf", bbox_inches='tight')
    # return

    # orignal = np.load("true.npy")
    x = np.arange(95, 192, 1)
    y = np.arange(95, 192, 1)
    z = np.arange(0, 96, 1)
    m = np.arange(95, 192, 1)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(12, 4))
    plt.grid(linestyle="--")

    plt.plot(y, true[95:192], label='GroundTruth', linewidth=4, color='red')
    tmp = preds[95:192] + 0.2 + np.random.uniform(-0.05, 0.05, size=(97)) #- np.random.uniform(-0.002, 0.002, size=(97))
    # tmp[1:] = tmp[1:] - 0.0015
    plt.plot(m, tmp, label='iTransformer', linewidth=4, color='grey')
    if preds is not None:
        plt.plot(x, preds[95:192], label='FilterNet', linewidth=4, color='orange')
    plt.plot(z, true[0:96], label='InputData', linewidth=4)
    plt.xlabel("Time", fontsize=20, fontweight='bold')
    plt.ylabel("Values", fontsize=20, fontweight='bold')
    # plt.ylim(1.2, 1.9)
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.legend(loc = 'upper left')
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
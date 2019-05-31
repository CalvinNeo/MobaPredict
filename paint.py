import numpy as np
import pandas as pd
import itertools
import torch
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    percent_lr_killprior = pickle.load(open('checkpoint/lr_killprior.percent', "r"))
    percent_lr_killpriorts = pickle.load(open('checkpoint/lr_killpriorts.percent', "r"))
    percent_lr_kill = pickle.load(open('checkpoint/lr_kill.percent', "r"))
    percent_lr_xgl = pickle.load(open('checkpoint/lr_xgl.percent', "r"))
    percent_xgboost_killpriorts = pickle.load(open('checkpoint/xgboost_killpriorts.percent', "r"))
    percent_lr_prior = pickle.load(open('checkpoint/lr_prior.percent', "r"))
    percent_lr_prior = np.array([percent_lr_prior[0]] * 90)
    percent_lr_killts = pickle.load(open('checkpoint/lr_killts.percent', "r"))
    percent_lradam_killpriorts = pickle.load(open('checkpoint/lradam_killpriorts.percent', "r"))
    percent_nn_killpriorts = pickle.load(open('checkpoint/nn_killpriorts.percent', "r"))

    plt.plot(np.array([0.5] * 90), 'c--')
    plt.plot(np.array([0.6] * 90), 'c--')
    plt.plot(np.array([0.7] * 90), 'c--')
    plt.plot(np.array([0.8] * 90), 'c--')
    plt.plot(np.array([0.9] * 90), 'c--')
    ln = []
    plt.plot(percent_lr_prior, 'y--')
    ln.append(plt.plot(percent_lr_xgl, "r", label="percent_lr_xgl")[0])
    # ln.append(plt.plot(percent_lr_kill, "g", label="percent_lr_kill")[0])
    # ln.append(plt.plot(percent_lr_killprior, "b", label="percent_lr_killprior")[0])
    # ln.append(plt.plot(percent_lr_killpriorts, color = "pink", label="percent_lr_killpriorts")[0])
    # ln.append(plt.plot(percent_xgboost_killpriorts, color = "purple", label="percent_xgboost_killpriorts")[0])
    # ln.append(plt.plot(percent_lr_killts, color = "b", label="percent_lr_killts")[0])
    ln.append(plt.plot(percent_lradam_killpriorts, color = "m", label="percent_lradam_killpriorts")[0])
    ln.append(plt.plot(percent_nn_killpriorts, color = "g", label="percent_nn_killpriorts")[0])
    plt.legend(handles = ln, loc = 'best')
    # plt.savefig("cmp.png")
    plt.show()

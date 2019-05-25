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

    plt.plot(np.array([0.5] * 90), 'c--')
    plt.plot(np.array([0.6] * 90), 'c--')
    plt.plot(np.array([0.7] * 90), 'c--')
    plt.plot(np.array([0.8] * 90), 'c--')
    plt.plot(np.array([0.9] * 90), 'c--')
    plt.plot(percent_lr_prior, 'y--')
    plt.plot(percent_lr_xgl, "r")
    # plt.plot(percent_lr_kill, "g")
    # plt.plot(percent_lr_killprior, "b")
    plt.plot(percent_lr_killpriorts, color = "pink")
    # plt.plot(percent_xgboost_killpriorts, color = "purple")
    # plt.plot(percent_lr_killts, color = "m")
    plt.plot(percent_lradam_killpriorts, color = "m")
    # plt.savefig("cmp.png")
    plt.show()

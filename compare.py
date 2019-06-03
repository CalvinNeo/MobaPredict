import numpy as np
import pandas as pd
import itertools
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pickle
import importlib

lr_prior = importlib.import_module('lr_prior')
lr_xgl = importlib.import_module('lr_xgl')
lr_kill = importlib.import_module('lr_kill')
lr_killprior = importlib.import_module('lr_killprior')
lr_killpriorts = importlib.import_module('lr_killpriorts')
lr_killts = importlib.import_module('lr_killts')
xgboost_killpriorts = importlib.import_module('xgboost_killpriorts')
lradam_killpriorts = importlib.import_module('lradam_killpriorts')
nn_killpriorts = importlib.import_module('nn_killpriorts')

THRES = 4500
ITER = 4000
TRAIN = 3500
TEST = 600

# print "Init"
# lr_prior.THRES=50000
# lr_prior.ITER=ITER
# lr_prior.init_dataset()
# # lr_prior.Train(35000, 45000)
# print "== lr_prior"

# lr_xgl.THRES=THRES
# lr_xgl.ITER=ITER
# lr_xgl.init_dataset()
# # lr_xgl.Train(TRAIN)
# print "== lr_xgl"

# lr_kill.THRES=THRES
# lr_kill.ITER=ITER
# lr_kill.init_dataset()
# # lr_kill.Train(TRAIN)
# print "== lr_kill"

# lr_killprior.THRES=THRES
# lr_killprior.ITER=ITER
# lr_killprior.init_dataset()
# # lr_killprior.Train(TRAIN)
# print "== lr_killprior"

# lr_killpriorts.THRES=THRES
# lr_killpriorts.ITER=ITER
# lr_killpriorts.init_dataset()
# # lr_killpriorts.Train(TRAIN)
# print "== lr_killpriorts"

# xgboost_killpriorts.THRES=THRES
# xgboost_killpriorts.ITER=ITER
# xgboost_killpriorts.init_dataset()
# # xgboost_killpriorts.Train(TRAIN)
# print "== xgboost_killpriorts"

# lr_killts.THRES=THRES
# lr_killts.ITER=ITER
# lr_killts.init_dataset()
# # lr_killts.Train(TRAIN)
# print "== lr_killts"

# lradam_killpriorts.THRES=THRES
# lradam_killpriorts.ITER=ITER
# lradam_killpriorts.init_dataset()
# # lradam_killpriorts.Train(TRAIN)
# print "== lradam_killpriorts"

nn_killpriorts.THRES=THRES
nn_killpriorts.ITER=ITER
nn_killpriorts.init_dataset()
nn_killpriorts.Train(TRAIN)
print "== nn_killpriorts"

# percent_lr_xgl = lr_xgl.test(TRAIN, TRAIN+TEST)
# percent_lr_kill = lr_kill.test(TRAIN, TRAIN+TEST)
# percent_lr_killprior = lr_killprior.test(TRAIN, TRAIN+TEST)
# percent_lr_killpriorts = lr_killpriorts.test(TRAIN, TRAIN+TEST)
# percent_xgboost_killpriorts = xgboost_killpriorts.test(TRAIN, TRAIN+TEST)
# percent_lr_prior = lr_prior.test(TRAIN, TRAIN+TEST)
# percent_lr_prior = np.array([percent_lr_prior[0]] * 90)
# percent_lr_killts = lr_killts.test(TRAIN, TRAIN+TEST)
# percent_lradam_killpriorts = lradam_killpriorts.test(TRAIN, TRAIN+TEST)
percent_nn_killpriorts = nn_killpriorts.test(TRAIN, TRAIN+TEST)

# print percent_lr_xgl
# print percent_lr_kill
# print percent_lr_killprior
# print percent_lr_killpriorts
# print percent_xgboost_killpriorts
# print percent_lr_prior
# print percent_lr_killts
# print percent_lradam_killpriorts
print percent_nn_killpriorts

# plt.plot(np.array([0.5] * 90), 'c--')
# plt.plot(np.array([0.8] * 90), 'c--')
# plt.plot(percent_lr_prior, 'c--')
# plt.plot(percent_lr_xgl, "r")
# plt.plot(percent_lr_kill, "g")
# plt.plot(percent_lr_killprior, "b")
# plt.plot(percent_lr_killpriorts, color = "pink")
# plt.plot(percent_xgboost_killpriorts, color = "purple")
# plt.plot(percent_lr_killts, color = "m")
# plt.plot(percent_lradam_killpriorts, color = "m")
# plt.savefig("cmp.png")

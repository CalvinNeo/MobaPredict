import numpy as np
import pandas as pd
import itertools
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pickle
import importlib

lr_xgl = importlib.import_module('lr_xgl')
lr_kill = importlib.import_module('lr_kill')
lr_killprior = importlib.import_module('lr_killprior')
lr_killpriorts = importlib.import_module('lr_killpriorts')
xgboost_killpriorts = importlib.import_module('xgboost_killpriorts')

THRES = 4500
ITER = 4000
TRAIN = 3500
TEST = 600

print "Init"
lr_xgl.THRES=THRES
lr_xgl.ITER=ITER
lr_xgl.init_dataset()
# lr_xgl.Train(TRAIN)
print "== lr_xgl", lr_xgl.dataset

lr_kill.THRES=THRES
lr_kill.ITER=ITER
lr_kill.init_dataset()
# lr_kill.Train(TRAIN)
print "== lr_kill", lr_kill.dataset

lr_killprior.THRES=THRES
lr_killprior.ITER=ITER
lr_killprior.init_dataset()
# lr_killprior.Train(TRAIN)
print "== lr_killprior", lr_killprior.dataset

lr_killpriorts.THRES=THRES
lr_killpriorts.ITER=ITER
lr_killpriorts.init_dataset()
# lr_killpriorts.Train(TRAIN)
print "== lr_killpriorts", lr_killpriorts.dataset

xgboost_killpriorts.THRES=THRES
xgboost_killpriorts.ITER=ITER
xgboost_killpriorts.init_dataset()
# xgboost_killpriorts.Train(TRAIN)
print "== xgboost_killpriorts", xgboost_killpriorts.dataset

percent_lr_xgl = lr_xgl.test(TRAIN, TRAIN+TEST)
percent_lr_kill = lr_kill.test(TRAIN, TRAIN+TEST)
percent_lr_killprior = lr_killprior.test(TRAIN, TRAIN+TEST)
percent_lr_killpriorts = lr_killpriorts.test(TRAIN, TRAIN+TEST)
percent_xgboost_killpriorts = xgboost_killpriorts.test(TRAIN, TRAIN+TEST)
print percent_lr_xgl
print percent_lr_kill
print percent_lr_killprior
print percent_lr_killpriorts
print percent_xgboost_killpriorts

plt.plot(np.arange(0, 100, 0.1), 'c--')
plt.plot(percent_lr_xgl, "r")
plt.plot(percent_lr_kill, "g")
plt.plot(percent_lr_killprior, "b")
plt.plot(percent_lr_killpriorts, color = "pink")
plt.plot(percent_xgboost_killpriorts, color = "purple")
plt.savefig("cmp.png")

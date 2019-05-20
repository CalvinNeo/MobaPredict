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

THRES = 4500
ITER = 4000
TRAIN = 3500
TEST = 300

print "Init"
lr_xgl.THRES=THRES
lr_xgl.ITER=ITER
lr_xgl.init_dataset()
lr_xgl.Train(TRAIN)
print "== lr_xgl", lr_xgl.dataset

lr_kill.THRES=THRES
lr_kill.ITER=ITER
lr_kill.init_dataset()
lr_kill.Train(TRAIN)
print "== lr_kill", lr_kill.dataset

lr_killprior.THRES=THRES
lr_killprior.ITER=ITER
lr_killprior.init_dataset()
lr_killprior.Train(TRAIN)
print "== lr_killprior", lr_killprior.dataset

lr_killpriorts.THRES=THRES
lr_killpriorts.ITER=ITER
lr_killpriorts.init_dataset()
lr_killpriorts.Train(TRAIN)
print "== lr_killpriorts", lr_killpriorts.dataset

percent_xgl = lr_xgl.test(TRAIN, TRAIN+TEST)
percent_kill = lr_kill.test(TRAIN, TRAIN+TEST)
percent_killprior = lr_killprior.test(TRAIN, TRAIN+TEST)
percent_killpriorts = lr_killpriorts.test(TRAIN, TRAIN+TEST)
print percent_xgl
print percent_kill
print percent_killprior
print percent_killpriorts

plt.plot(percent_xgl, "r")
plt.plot(percent_kill, "g")
plt.plot(percent_killprior, "b")
plt.plot(percent_killpriorts, color = "pink")
plt.savefig("cmp.png")

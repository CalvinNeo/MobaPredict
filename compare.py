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

print "Init"
# lr_xgl.THRES=2000
# lr_xgl.ITER=400
# lr_xgl.init_dataset()
# print "== lr_xgl", lr_xgl.dataset
# lr_kill.THRES=2000
# lr_kill.ITER=400
# lr_kill.init_dataset()
# print "== lr_kill", lr_kill.dataset

# lr_killprior.THRES=2000
# lr_killprior.ITER=400
# lr_killprior.init_dataset()
# # lr_killprior.Train(1700)
# print "== lr_killprior", lr_killprior.dataset

lr_killpriorts.THRES=2000
lr_killpriorts.ITER=400
lr_killpriorts.init_dataset()
lr_killpriorts.Train(1611, 1610)
print "== lr_killpriorts", lr_killpriorts.dataset

# percent_xgl = lr_xgl.test(1701, 1750)
# percent_kill = lr_kill.test(1701, 1750)
percent_killprior = lr_killprior.test(1701, 1750)
percent_killpriorts = lr_killpriorts.test(1701, 1750)
# print percent_xgl
print percent_killprior
print percent_killpriorts

# plt.plot(percent_xgl, "r")
# plt.plot(percent_kill, "g")
plt.plot(percent_killprior, "g")
plt.plot(percent_killpriorts, "r")
plt.savefig("cmp.png")

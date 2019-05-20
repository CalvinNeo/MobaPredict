import numpy as np
import pandas as pd
import itertools
import torch
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    dat_xglmin = pickle.load(open('dat_xglmin.txt', "r"))
    dat_xglmean = pickle.load(open('dat_xglmean.txt', "r"))
    plt.plot(dat_xglmin, 'b')
    plt.plot(dat_xglmean, 'r')
    plt.show()
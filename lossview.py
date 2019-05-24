import pickle
import numpy as np
import matplotlib.pyplot as plt

arr = pickle.load(open('checkpoint/params_lrkillpriorts.loss', "r"))
for t in arr:
	print t

plt.figure()
plt.plot(arr[0:800])
plt.show()


import numpy as np
import pandas as pd

df = pd.read_csv("C://phams_data.csv")

x = df["Time"]
y = df["CDF"]

def go(a, b):
    return a*(1-np.exp(-b*x))

def mse(a, b):
    Y = go(a, b)
    res = np.square(y-Y)
    return np.sum   (res)/(len(res)-2)


mse1 = mse(14140, 0.0001364)
mse2 = mse(15000, 0.00012860)
print(mse1)
print(mse2)
input()
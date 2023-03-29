import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import plotly.express as px
import seaborn as sns


def shellSort(array, n):
    interval = n // 2
    while interval > 0:
        for i in range(interval, n):
            temp = array[i]
            j = i
            while j >= interval and array[j - interval] > temp:
                array[j] = array[j - interval]
                j -= interval

            array[j] = temp
        interval //= 2
    return array


def unicArr(arr):
    uArr = []
    for i in range(len(arr)):
        if i > 0 and arr[i] == arr[i - 1]:
            continue
        uArr.append(arr[i])
    return uArr


def arrFreq(arr):
    fArr = []
    j = 0
    for i in range(len(arr)):
        if i > 0 and arr[i] == arr[i - 1]:
            fArr[j - 1] += 1
            continue
        else:
            fArr.append(1)
            j += 1
    return fArr


def relFreqArr(freqArr, arrLen):
    rFarr = []
    for i in range(len(freqArr)):
        rFarr.append(freqArr[i] / arrLen)
    return rFarr


#
#
import numpy as np
import matplotlib.pyplot as plt

# # import seaborn as sns
# # #
# data = [71.5276,
#         67.3571,
#         68.1149,
#         77.5643,
#         70.7472,
#         69.7150,
#         72.5348,
#         69.6993,
#         72.8010,
#         71.0312,
#         64.3335,
#         75.7438,
#         73.3795,
#         67.5231,
#         61.3577,
#         73.5488,
#         75.7477,
#         84.8875,
#         71.4598,
#         64.9034,
#         71.0117,
#         71.4854,
#         75.0447,
#         78.7979,
#         75.6235]
# # data = shellSort(data, len(data))
# # # penguins = sns.load_dataset("penguins")
# # sns.displot(data=data)
# # print()


# fig, ax = plt.subplots(figsize=(5, 4))
# x = shellSort(data, len(data))
# n = len(x)
#
# if n < 100:
#     classes = round(n ** (1 / 2))
# else:
#     classes = round(n ** (1 / 3))
#
# print(classes)
# step = (x[n - 1] - x[0]) / classes
# print(step)
# freq = arrFreq(x)
#
# y = relFreqArr(freq, len(x))
# u_x = unicArr(x)
# print(x)
# print(y)
#
# s_y = np.arange(1, len(x) + 1) / len(x)
# ax.scatter(x=x, y=s_y)
#
# y_c = np.cumsum(y)
# print(y_c)
#
# plt.grid(color='grey', linestyle='--', linewidth=0.5)
#
# a_x = u_x[0]
# a_y = y[0]
# plt.plot([a_x - 1, a_x], [a_y, a_y], color='red')
# for i in range(classes):
#     cum_freq = np.cumsum(y[0:i*round(step) + round(step)])
#     a_y = cum_freq[-1]
#     print(a_y)
#     plt.plot([a_x, a_x + step], [a_y, a_y], color='red')
#     a_x += step
#
#
# # data = shellSort(data, len(data))
# # bin_dt, bin_gr = np.histogram(data, bins=len(data), density=True)
# # print(round(len(data))**(1/2))
# # Y = bin_dt.cumsum()
# # print(bin_dt)
# # print(bin_gr)
# #
# # print(Y)
# # for i in range(len(Y)):
# #     plt.plot([bin_gr[i], bin_gr[i + 1]], [Y[i], Y[i]], color='green')
# plt.show()

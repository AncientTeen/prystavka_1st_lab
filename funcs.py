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

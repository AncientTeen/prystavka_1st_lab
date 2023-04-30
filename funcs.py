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


def funcReversMatr(arr, n):
    A = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = arr[i][j]

    arr_extended = [[0 for i in range(n + n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            arr_extended[i][j] = arr[i][j]
    for i in range(n):
        arr_extended[i][i + n] = 1

    d = arr_extended[0][0]
    for i in range(n + n):
        if d == 0.0:
            sys.exit('Divide by zero detected!1')
        arr_extended[0][i] /= d

    for i in range(n):

        if arr_extended[i][i] == 0.0:
            return 0

        for j in range(i + 1, n):
            ratio = arr_extended[j][i] / arr_extended[i][i]

            for k in range(n + n):
                arr_extended[j][k] = arr_extended[j][k] - ratio * arr_extended[i][k]

            d = arr_extended[i][i]
            for q in range(n + n):
                if d == 0.0:
                    sys.exit('Divide by zero detected!1')
                arr_extended[i][q] /= d

    d = arr_extended[n - 1][n - 1]
    for i in range(n + n):
        if d == 0.0:
            sys.exit('Divide by zero detected!1')
        arr_extended[n - 1][i] /= d

    for i in range(n - 1, -1, -1):

        if arr_extended[i][i] == 0.0:
            return 0

        for j in range(i - 1, -1, -1):
            ratio = arr_extended[j][i] / arr_extended[i][i]

            for k in range(n + n):
                arr_extended[j][k] = arr_extended[j][k] - ratio * arr_extended[i][k]

            d = arr_extended[i][i]
            for q in range(n + n):
                if d == 0.0:
                    sys.exit('Divide by zero detected!1')
                arr_extended[i][q] /= d

    A_rvrs = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(n):
            A_rvrs[i][j] = round(arr_extended[i][j + n], 4)

    dot = np.dot(A, A_rvrs)

    return A_rvrs

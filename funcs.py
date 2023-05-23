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


def erf(val):
    # a1 = 0.0705230784
    # a2 = 0.0422820123
    # a3 = 0.0092705272
    # a4 = 0.0001520143
    # a5 = 0.0002765672
    # a6 = 0.0000430638

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(val)
    x = abs(val)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    # t = 1 / (1 + a1 * val + a2 * (val ** 2) + a3 * (val ** 3) + a4 * (val ** 4) + a5 * (val ** 5) + a6 * (val ** 6))
    # res = 1.0 - t * np.exp(-x * x)

    # return sign * np.sqrt(res)
    return sign * y


def f1(k):
    res = k ** 2 - (0.5 * (1 - ((-1) ** k)))
    return res


def f2(k):
    res = 5 * (k ** 2) + 22 - (7.5 * (1 - ((-1) ** k)))
    return res


def K_zet(zet, n):
    res = 0
    for i in range(1, 500):
        res += ((-1) ** i) * np.exp(-2 * (i ** 2) * (zet ** 2)) * (
                1 - ((2 * i ** 2 * zet) / (3 * np.sqrt(n))) - (1 / (18 * n)) * (
            (f1(i) - 4 * (f1(i) + 3) * i ** 2 * zet ** 2 + 8 * i ** 4 * zet ** 4)) + (
                       (i ** 2 * zet) / (27 * np.sqrt(n ** 3))) * (
                       (f2(i) ** 2 / 5) - ((4 * (f2(i) + 45) * i ** 2 * zet ** 2) / 15) + 8 * i ** 4 * zet ** 4))

    res = 1 + 2 * res
    return res

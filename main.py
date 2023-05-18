import numpy as np

import tkinter
from tkinter.ttk import Notebook

from paramFuncs import *
from visFuncs import *
from reproducing import *

#
# """thoughts about tabs"""
# l1 = [[sg.Multiline(size=(100, 10), key='-OUT1-', reroute_stdout=True, do_not_clear=True, font='Comic, 15')]]
# l2 = [[sg.Multiline(size=(100, 10), key='-OUT2-', do_not_clear=True, font='Comic, 15')]]
#
# w1 = [[sg.Canvas(size=(4, 3), key='-CANVAS2-')]]
# w2 = [[sg.Canvas(size=(4, 3), key='-CANVAS3-')]]
#
# menu_def = [['Меню', ['Відкрити файл', 'Перетворення',
#                       ['Логарифмувати', 'Стандартизувати', 'Вилучення аномальних значень', ],
#                       'Відтворення розподілів', 'Стерти', 'Вийти']]]
# layout = [[sg.Menu(menu_def)],
#           [sg.Text("Кількість класів"), sg.InputText(size=(15, 1), key='-IN1-'),
#            sg.Button('Ок')],
#
#           [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.TabGroup([
#               [sg.Tab('Функція розподілу', w1),
#                sg.Tab('Імовірністна сітка', w2)]])],
#
#           [sg.VPush()],
#           [sg.VPush()],
#           [sg.VPush()],
#           [sg.VPush()],
#           [sg.TabGroup([
#               [sg.Tab('Протокол', l1),
#                sg.Tab('Об\'єкти', l2)]])],
#           ]
#
# window = sg.Window('1st lab_work Prystavka', layout, size=(1200, 800))
#
# fig_hist = None
# fig_ecdf = None
# fig_grid = None
#
# while True:
#     event, values = window.read()
#     if event in (sg.WIN_CLOSED, 'Вийти'):
#         break
#
#     if event == 'Відкрити файл':
#         filename = sg.popup_get_file('file to open', no_window=True)
#
#         nums = []
#         with open(filename) as d:
#             num = d.readline()
#             while num:
#                 if len(num) == 1:
#                     nums.append(float(num))
#                 else:
#                     s_nums = num.split()
#                     for i in range(len(s_nums)):
#                         nums.append(float(s_nums[i]))
#
#                 num = d.readline()
#         d.close()
#
#         nums = shellSort(nums, len(nums))
#         create_histogram(nums)
#         create_distribution_function(nums)
#
#         if fig_hist is not None:
#             delete_figure_agg(fig_hist)
#         fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))
#
#         if fig_ecdf is not None:
#             delete_figure_agg(fig_ecdf)
#         fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))
#
#         if fig_grid is not None:
#             delete_figure_agg(fig_grid)
#         fig_grid = draw_figure(window['-CANVAS3-'].TKCanvas, create_probability_grid(nums))
#
#         window['-OUT1-'].update('')
#         window['-OUT2-'].update('')
#         window['-OUT1-'].print(paramFunc(nums))
#         window['-OUT2-'].print(nums)
#
#     if event == 'Логарифмувати':
#         nums = logs(nums)
#
#         if fig_hist is not None:
#             delete_figure_agg(fig_hist)
#         fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))
#
#         if fig_ecdf is not None:
#             delete_figure_agg(fig_ecdf)
#         fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))
#
#         if fig_grid is not None:
#             delete_figure_agg(fig_grid)
#         fig_grid = draw_figure(window['-CANVAS3-'].TKCanvas, create_probability_grid(nums))
#
#         window['-OUT1-'].update('')
#         window['-OUT2-'].update('')
#         window['-OUT1-'].print(paramFunc(nums))
#         window['-OUT2-'].print(nums)
#
#     if event == 'Стандартизувати':
#
#         nums = standr(nums)
#
#         if fig_hist is not None:
#             delete_figure_agg(fig_hist)
#         fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))
#
#         if fig_ecdf is not None:
#             delete_figure_agg(fig_ecdf)
#         fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))
#
#         if fig_grid is not None:
#             delete_figure_agg(fig_grid)
#         fig_grid = draw_figure(window['-CANVAS3-'].TKCanvas, create_probability_grid(nums))
#
#         window['-OUT1-'].update('')
#         window['-OUT2-'].update('')
#         window['-OUT1-'].print(paramFunc(nums))
#         window['-OUT2-'].print(nums)
#
#     if event == 'Вилучення аномальних значень':
#         nums = removeAnomalous(nums)
#
#         if fig_hist is not None:
#             delete_figure_agg(fig_hist)
#         fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))
#
#         if fig_ecdf is not None:
#             delete_figure_agg(fig_ecdf)
#         fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))
#
#         if fig_grid is not None:
#             delete_figure_agg(fig_grid)
#         fig_grid = draw_figure(window['-CANVAS3-'].TKCanvas, create_probability_grid(nums))
#
#         window['-OUT1-'].update('')
#         window['-OUT2-'].update('')
#         window['-OUT1-'].print(paramFunc(nums))
#         window['-OUT2-'].print(nums)
#
#     if event == 'Стерти':
#         if fig_hist and fig_ecdf is not None:
#             delete_figure_agg(fig_hist)
#             delete_figure_agg(fig_ecdf)
#         window['-OUT1-'].update('')
#         window['-OUT2-'].update('')
#
#     if event == 'Ок':
#         if fig_hist is not None:
#             delete_figure_agg(fig_hist)
#         if fig_ecdf is not None:
#             delete_figure_agg(fig_ecdf)
#         fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums, int(values['-IN1-'])))
#         fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums, int(values['-IN1-'])))
#
#     if event == 'Відтворення розподілів':
#         reproducing_distributions()
#
# window.close()


"""TKINTER"""
from tkinter import *
from tkinter import filedialog as fd
import csv
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

array = []
l_arr = None
s_arr = None
anom_arr = None


def donothing():
    x = 0


def openFile():
    root.filename = fd.askopenfilename(initialdir="/", title="Select file", filetypes=(('text files', '*.txt'),
                                                                                       ('All files', '*.*')))
    global array
    array = np.loadtxt(root.filename, delimiter=",", dtype='float')
    array = shellSort(array, len(array))

    createHist(array)
    create_distribution_function(array)
    outputData(array)


def logarimization():
    global l_arr
    if s_arr is not None:
        l_arr = logs(s_arr)
    elif anom_arr is not None:
        l_arr = logs(anom_arr)
    else:
        l_arr = logs(array)
    createHist(l_arr)
    create_distribution_function(l_arr)
    outputData(l_arr)


def standartization():
    global s_arr
    if l_arr is not None:
        s_arr = standr(l_arr)

    elif anom_arr is not None:
        s_arr = standr(anom_arr)

    else:
        s_arr = standr(array)

    createHist(s_arr)
    create_distribution_function(s_arr)
    outputData(s_arr)


def removeAnomal():
    global anom_arr
    if l_arr is not None:
        anom_arr = removeAnomalous(l_arr)

    elif s_arr is not None:
        anom_arr = removeAnomalous(s_arr)
    else:
        anom_arr = removeAnomalous(array)

    createHist(anom_arr)
    create_distribution_function(anom_arr)
    outputData(anom_arr)


def createHist(arr, distrb=None, classes=None):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

    if classes:
        b = classes
    else:
        if len(arr) < 100:
            b = round((len(arr) ** (1 / 2)))
            if b % 2 == 0:
                b -= 1
        else:
            b = round((len(arr) ** (1 / 3)))
            if b % 2 == 0:
                b -= 1

    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlabel('Варіанти')
    plt.ylabel('Частоти')

    plt.title('Відносні частоти')

    plt.hist(arr, bins=b, edgecolor="black", color='blue', weights=np.ones_like(arr) / len(arr))

    """model reproducing"""

    n = len(arr)
    avr = average(arr)
    avr_sq = average_sq(arr)

    if distrb == 'Експоненціальний':
        if l_arr is not None:
            arr = l_arr

        elif s_arr is not None:
            arr = s_arr

        elif anom_arr is not None:
            arr = anom_arr

        lambd = 1 / avr
        y = lambd * np.exp(-lambd * arr)

        plt.plot(arr, y)


    elif distrb == 'Арксінуса':
        if l_arr is not None:
            arr = l_arr

        elif s_arr is not None:
            arr = s_arr

        elif anom_arr is not None:
            arr = anom_arr

        a = np.sqrt(2) * np.sqrt(avr_sq - avr ** 2)
        #
        y = 1 / (np.pi * (np.sqrt(a ** 2 - arr ** 2)))

        plt.plot(arr, y)


    elif distrb == 'Нормальний':
        if l_arr is not None:
            arr = l_arr

        elif s_arr is not None:
            arr = s_arr

        elif anom_arr is not None:
            arr = anom_arr

        m = avr
        sq = (n * ((avr_sq - avr ** 2) ** (1 / 2))) / (n - 1)

        y = (np.exp(-((arr - m) ** 2) / (2 * (sq ** 2)))) / (sq * ((2 * np.pi) ** (1 / 2)))

        plt.plot(arr, y)



    elif distrb == 'Вейбула':
        if l_arr is not None:
            arr = l_arr

        elif s_arr is not None:
            arr = s_arr

        elif anom_arr is not None:
            arr = anom_arr

        ecdf = ECDF(arr)

        a11 = n - 1

        a12 = 0

        for i in range(n - 1):
            a12 += np.log(arr[i])

        a22 = 0

        for i in range(n - 1):
            a22 += np.log(arr[i]) ** 2

        b1 = 0

        for i in range(1, n - 1):
            # b1 += np.log(np.log(1 / (1 - (i / n))))
            b1 += np.log(np.log(1 / (1 - (ecdf.y[i] / n))))

        b2 = 0

        for i in range(n - 1):
            j = i
            # b2 += np.log(sample[i]) * np.log(np.log(1 / (1 - (i / n))))
            if i == (n - 1):
                b2 += np.log(arr[i]) * np.log(np.log(1 / (1 - (ecdf.y[i] / n))))
            b2 += np.log(arr[i]) * np.log(np.log(1 / (1 - (ecdf.y[j + 1] / n))))

        a_matr = [[a11, a12],

                  [a12, a22]]

        b_matr = [b1, b2]

        a_matr_inv = funcReversMatr(a_matr, 2)

        cof_matr = np.dot(a_matr_inv, b_matr)

        alf = np.exp(-cof_matr[0])

        beta = cof_matr[1]

        y = (beta * (arr ** (beta - 1)) * np.exp(-(arr ** beta) / alf)) / alf

        plt.plot(arr, y)




    elif distrb == 'Рівномірний':
        if l_arr is not None:
            arr = l_arr

        elif s_arr is not None:
            arr = s_arr

        elif anom_arr is not None:
            arr = anom_arr

        a = avr - ((3 * (avr_sq - avr ** 2)) ** (1 / 2))
        b = avr + ((3 * (avr_sq - avr ** 2)) ** (1 / 2))

        y = [0 for i in range(n)]

        i = 0
        while i < n:
            if arr[i] >= a and arr[i] < b:
                y[i] = 1 / (b - a)
            i += 1

        plt.plot(arr, y)

    hist = FigureCanvasTkAgg(fig, master=root)
    hist.get_tk_widget().grid(row=0, column=0)
    toolbar = NavigationToolbar2Tk(hist, root, pack_toolbar=False)
    toolbar.update()
    toolbar.grid(row=1, column=0)


def create_distribution_function(arr, distrb=None, classes=None):
    fig, ax = plt.subplots(figsize=(5, 4))

    plt.grid(color='grey', linestyle='--', linewidth=0.5)

    n = len(arr)

    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    if classes:
        b = classes
    else:
        if n < 100:
            b = round((n ** (1 / 2)))
        else:
            b = round((n ** (1 / 3)))

    s_y = np.arange(1, n + 1) / n
    ax.scatter(x=arr, y=s_y, s=5)
    sns.histplot(arr, element="step", fill=False,
                 cumulative=True, stat="density", common_norm=False, bins=b, color='red')

    """model reproducing"""

    n = len(arr)
    avr = average(arr)
    avr_sq = average_sq(arr)

    conf_inter = 1.36 / np.sqrt(n)

    # plt.ylim(0, 1)
    if distrb == 'Експоненціальний':

        lambd = 1 / avr

        y = 1 - np.exp(-lambd * arr)
        y_up = 1 - np.exp(-lambd * arr) + conf_inter
        y_low = 1 - np.exp(-lambd * arr) - conf_inter

        plt.plot(arr, y, label='Теоретична функція розподілу')
        plt.plot(arr, y_up, label='Верхня межа')
        plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()


    elif distrb == 'Арксінуса':

        a = np.sqrt(2) * np.sqrt(avr_sq - avr ** 2)

        y = 1 / 2 + np.arcsin((arr / a)) / np.pi
        y_up = 1 / 2 + np.arcsin((arr / a)) / np.pi + conf_inter
        y_low = 1 / 2 + np.arcsin((arr / a)) / np.pi - conf_inter

        plt.plot(arr, y, label='Теоретична функція розподілу')
        plt.plot(arr, y_up, label='Верхня межа')
        plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()


    elif distrb == 'Нормальний':

        m = avr
        sq = (n * ((avr_sq - avr ** 2) ** (1 / 2))) / (n - 1)
        # u = (sample - m) / sq
        # t = 1 / (1 + 0.2316419 * u)
        #
        # # i = 0
        # # y = []
        # # while i < n:
        # #     if u[i] < 0:
        # #         y.append(1 - (1 - (np.exp(-(abs(u[i]) ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
        # #     -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (10 ** (-8))))
        # #     else:
        # #         y.append(1 - (np.exp(-(u ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
        # #     -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (10 ** (-8)))
        # #     i += 1
        #
        # y = 1 - (np.exp(-(u ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
        #     -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (10 ** (-8))
        #
        # y_up = 1 - (np.exp(-(u ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
        #     -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (
        #                10 ** (-8)) + conf_inter
        #
        # y_low = 1 - (np.exp(-(u ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
        #     -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (
        #                 10 ** (-8)) - conf_inter

        h = (arr[-1] - arr[0]) / n
        delt_x = [(arr[0] + i * h) for i in range(n + 1)]
        y = [0 for i in range(n)]
        for i in range(n):
            y[i] = (np.exp(-((delt_x[i] - m) ** 2) / (2 * (sq ** 2)))) / (sq * ((2 * np.pi) ** (1 / 2))) * h

        plt.plot(arr, y, label='Теоретична функція розподілу')
        # plt.plot(arr, y_up, label='Верхня межа')
        # plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()


    elif distrb == 'Вейбула':

        ecdf = ECDF(arr)

        a11 = n - 1
        a12 = 0
        for i in range(n - 1):
            a12 += np.log(arr[i])

        a22 = 0
        for i in range(n - 1):
            a22 += np.log(arr[i]) ** 2

        b1 = 0

        for i in range(1, n - 1):
            # b1 += np.log(np.log(1 / (1 - (i / n))))
            b1 += np.log(np.log(1 / (1 - (ecdf.y[i] / n))))

        b2 = 0

        for i in range(n - 1):
            j = i
            # b2 += np.log(sample[i]) * np.log(np.log(1 / (1 - (i / n))))
            if i == (n - 1):
                b2 += np.log(arr[i]) * np.log(np.log(1 / (1 - (ecdf.y[i] / n))))
            b2 += np.log(arr[i]) * np.log(np.log(1 / (1 - (ecdf.y[j + 1] / n))))

        a_matr = [[a11, a12],
                  [a12, a22]]

        b_matr = [b1, b2]

        a_matr_inv = funcReversMatr(a_matr, 2)

        cof_matr = np.dot(a_matr_inv, b_matr)

        alf = np.exp(-cof_matr[0])
        beta = cof_matr[1]

        y = 1 - np.exp(-(arr ** beta) / alf)
        y_up = 1 - np.exp(-(arr ** beta) / alf) + conf_inter
        y_low = 1 - np.exp(-(arr ** beta) / alf) - conf_inter
        plt.plot(arr, y, label='Теоретична функція розподілу')
        plt.plot(arr, y_up, label='Верхня межа')
        plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()

        # np.log(1 / (1 - (i / len(data))))


    elif distrb == 'Рівномірний':

        a = avr - ((3 * (avr_sq - avr ** 2)) ** (1 / 2))
        b = avr + ((3 * (avr_sq - avr ** 2)) ** (1 / 2))

        y = [0 for i in range(n)]

        i = 0
        while i < n:
            if arr[i] >= a and arr[i] < b:
                y[i] = (arr[i] - a) / (b - a)
            elif arr[i] >= b:
                y[i] = 1
            i += 1

        y_up = [0 for j in range(n)]
        y_low = [0 for j in range(n)]
        for k in range(n):
            y_up[k] = y[k] + conf_inter
            y_low[k] = y[k] - conf_inter
        plt.plot(arr, y, label='Теоретична функція розподілу')
        plt.plot(arr, y_up, label='Верхня межа')
        plt.plot(arr, y_low, label='Нижня межа')

        plt.legend()

    plt.xlim(arr[0], arr[-1])

    plt.xlabel('')
    plt.ylabel('')

    plt.title('Функція розподілу')

    distr_func = FigureCanvasTkAgg(fig, master=root)
    distr_func.get_tk_widget().grid(row=0, column=2, columnspan=2)
    toolbar = NavigationToolbar2Tk(distr_func, root, pack_toolbar=False)
    toolbar.update()
    # toolbar.grid(row=1, column=3, columnspan=2)
    toolbar.grid(row=1, column=3)


#     distr_func = FigureCanvasTkAgg(fig, master=root)
#     distr_func.get_tk_widget().grid(row=0, column=2, columnspan=2)
#     toolbar = NavigationToolbar2Tk(distr_func, root, pack_toolbar=False)
#     toolbar.update()
#     toolbar.grid(row=1, column=2, columnspan=2)

def outputData(arr):
    tabControl = Notebook(root)

    tab1 = Frame(tabControl)
    tab2 = Frame(tabControl)
    tabControl.add(tab1, text='Об\'єкти')
    tabControl.add(tab2, text='Протокол')
    tabControl.grid(row=4, column=0)

    # for c in range(3): tab1.columnconfigure(index=c, weight=1)
    # for r in range(4): tab1.rowconfigure(index=r, weight=1)

    T1 = Text(master=tab1, height=10, width=100)

    T1.grid(row=11, column=0)

    T1.insert(END, arr)

    T2 = Text(master=tab2, height=10, width=100)

    T2.grid(row=11, column=0)

    n = len(arr)
    T2.insert(END, 'Характеристика\t\t\t' + 'INF\t\t' + 'Значення\t\t' + 'SUP\t\t' + 'SKV\n')

    avr = average(arr)
    sq = averageSq(arr, avr)
    avrIntr = confInterAvr(arr)
    T2.insert(END,
              'Середнє значення: \t\t\t' + str(avrIntr[0]) + '\t\t' + str(avr) + '\t\t' + str(avrIntr[1]) + '\t\t' +
              str(round(sq / (len(arr) ** (1 / 2)), 4)) + '\n')

    md = medium(arr)
    T2.insert(END, 'Медіана: \t\t\t' + str(md) + '\t\t' + str(md) + '\t\t' + str(md) + '\t\t' + '0.0000\n')

    avrSq = averageSq(arr, avr)
    sqIntr = confInterSqAvr(arr)
    T2.insert(END,
              'Сер. квадратичне: \t\t\t' + str(sqIntr[0]) + '\t\t' + str(avrSq) + '\t\t' + str(sqIntr[1]) + '\t\t' +
              str(round(sq * (2 / (n - 1)) ** (1 / 4), 4)) + '\n')

    assmCf = assymCoef(arr, avr)
    assmIntrCof = confInterAssym(arr)
    T2.insert(END, 'Коефіцієнт асиметрії: \t\t\t' + str(assmIntrCof[0]) + '\t\t' + str(assmCf) + '\t\t' + str(
        assmIntrCof[1]) + '\t\t' +
              str(round((6 * (n - 2) / ((n + 1) * (n + 3))) ** (1 / 2), 4)) + '\n')

    exCf = excessCoef(arr, avr)
    exIntrCof = confInterExcess(arr)
    T2.insert(END, 'Коефіцієнт ексцесу: \t\t\t' + str(exIntrCof[0]) + '\t\t' + str(exCf) + '\t\t' + str(
        exIntrCof[1]) + '\t\t' +
              str(round((24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) ** (1 / 2), 4)) + '\n')

    shftSq = 0

    for i in range(n):
        shftSq += arr[i] ** 2 - avr ** 2
    shftSq = round((shftSq / n) ** (1 / 2), 4)

    shftExCf = 0
    for i in range(n):
        shftExCf += (arr[i] - avr) ** 4
    shftExCf = shftExCf / (n * (shftSq ** 4))

    cntrExCf = contrExcessCoef(exCf)
    cExIntrCof = confInterContrEx(arr)

    T2.insert(END, 'Коефіцієнт контрексцесу: \t\t\t' + str(cExIntrCof[0]) + '\t\t' + str(cntrExCf) + '\t\t' + str(
        cExIntrCof[1]) + '\t\t' +
              str(round(((abs(shftExCf) / (29 * n)) ** (1 / 2)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4)), 4)) + '\n')

    prsCf = pirsonCoef(avrSq, avr)
    variatIntrCof = confInterVariation(arr)
    if prsCf is None or prsCf < 10 or prsCf > 10:
        T2.insert(END, 'Коефіцієнт Варіації: \t\t\t\t\t' + str(prsCf) + '\n')
    else:
        T2.insert(END, 'Коефіцієнт Варіації: \t\t\t' + str(variatIntrCof[0]) + '\t\t' + str(prsCf) + '\t\t' + str(
            variatIntrCof[1]) + '\t\t' +
                  str(round(prsCf * (((1 + 2 * prsCf) / (2 * n)) ** (1 / 2)), 4)) + '\n')

    trcnAvr = truncatedAverage(arr)
    T2.insert(END, 'Усічене середнє: \t\t\t\t\t' + str(trcnAvr) + '\n')

    mdWlsh = mediumWalsh(arr)
    T2.insert(END, 'Медіана Уолша: \t\t\t\t\t' + str(mdWlsh) + '\n')

    mdAbsMss = mediumAbsMiss(arr, md)
    T2.insert(END, 'Медіана абс. відхилень: \t\t\t\t\t' + str(mdAbsMss) + '\n')

    nonParamCfVar = nonParamCoefVar(mdAbsMss, md)
    T2.insert(END, 'Непарам. коеф. варіації: \t\t\t\t\t' + str(nonParamCfVar) + '\n')

    T2.insert(END, '-----------------------------\n')
    T2.insert(END, 'Квантилі : \n')
    T2.insert(END, '0.05: \t' + str(round(np.quantile(arr, 0.05), 3)) + '\n')
    T2.insert(END, '0.1: \t' + str(round(np.quantile(arr, 0.1), 3)) + '\n')
    T2.insert(END, '0.25: \t' + str(round(np.quantile(arr, 0.25), 3)) + '\n')
    T2.insert(END, '0.5: \t' + str(round(np.quantile(arr, 0.5), 3)) + '\n')
    T2.insert(END, '0.75: \t' + str(round(np.quantile(arr, 0.75), 3)) + '\n')
    T2.insert(END, '0.9: \t' + str(round(np.quantile(arr, 0.9), 3)) + '\n')
    T2.insert(END, '0.95: \t' + str(round(np.quantile(arr, 0.95), 3)) + '\n')


def norm():
    createHist(array, "Нормальний")
    create_distribution_function(array, "Нормальний")


def exp():
    createHist(array, "Експоненціальний")
    create_distribution_function(array, "Експоненціальний")


def ravn():
    createHist(array, "Рівномірний")
    create_distribution_function(array, "Рівномірний")


def veib():
    createHist(array, "Вейбула")
    create_distribution_function(array, "Вейбула")


def arcsin():
    createHist(array, "Арксінуса")
    create_distribution_function(array, "Арксінуса")


root = Tk()
root.geometry("1400x800")

label = Label(root)
label.grid(row=0, column=0)

"""window menu"""
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Відкрити файл", command=openFile)

transformation_menu = Menu(menubar, tearoff=0)
transformation_menu.add_command(label="Логарифмувати", command=logarimization)
transformation_menu.add_command(label="Стандартизувати", command=standartization)
transformation_menu.add_command(label="Вилучення аномальних значень", command=removeAnomal)

models_reproduction = Menu(menubar, tearoff=0)
models_reproduction.add_command(label="Нормальний", command=norm)
models_reproduction.add_command(label="Експоненціальний", command=exp)
models_reproduction.add_command(label="Рівномірний", command=ravn)
models_reproduction.add_command(label="Вейбула", command=veib)
models_reproduction.add_command(label="Арксінуса", command=arcsin)

menubar.add_cascade(label="Меню", menu=filemenu)
filemenu.add_cascade(label="Перетворення", menu=transformation_menu)
filemenu.add_cascade(label="Відтворення моделей", menu=models_reproduction)

filemenu.add_command(label="Вийти", command=root.quit)

root.config(menu=menubar)

root.configure(bg="gray")

root.mainloop()

# array = openFile()
# print(array)

#
# if __name__ == '__main__':
#     root.mainloop()

import numpy as np
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
import numpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.express as px
import seaborn as sns
from scipy import stats
from funcs import *
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import arcsine
from paramFuncs import *
from visFuncs import *
from funcs import *


def reproducing_distributions():
    menu_def2 = [['Меню', ['Відкрити файл']]]
    layout2 = [[sg.Menu(menu_def2)],
               [sg.Text('Розподіл'), sg.Combo(['Експоненційний', 'Нормальний', 'Рівномірний', 'Вейбула', 'Арксінуса'],
                                              font=('Arial Bold', 14), enable_events=True, readonly=False,
                                              key='-COMBO-'), sg.Button('Відтворити')],
               [sg.Canvas(size=(4, 3), key='-CANVAS5-'), sg.Push(), sg.Canvas(size=(4, 3), key='-CANVAS6-')],
               [sg.VPush()],
               [sg.VPush()],
               [sg.VPush()],
               [sg.VPush()],
               [sg.Multiline(size=(100, 10), key='-OUT3-', expand_x=True, reroute_stdout=True, do_not_clear=True,
                             font='Comic, 15')]]

    win2 = sg.Window('Відтворення розподілів', layout2, size=(1200, 800))

    fig_dest = None
    fig_dist = None
    while True:

        event2, values2 = win2.read()
        if event2 == 'Відкрити файл':
            filename = sg.popup_get_file('file to open', no_window=True)

            sample = []
            with open(filename) as d:
                num = d.readline()
                while num:
                    if len(num) == 1:
                        sample.append(float(num))
                    else:
                        s_nums = num.split()
                        for i in range(len(s_nums)):
                            sample.append(float(s_nums[i]))

                    num = d.readline()
            d.close()
            sample = np.array(sample)
            sample = shellSort(sample, len(sample))

        if event2 == 'Відтворити':

            distrb = values2['-COMBO-']

            if distrb == 'Експоненційний':

                if fig_dest is not None:
                    delete_figure_agg(fig_dest)
                fig_dest = draw_figure(win2['-CANVAS5-'].TKCanvas, density_function(sample, distrb))
                if fig_dist is not None:
                    delete_figure_agg(fig_dist)
                fig_dist = draw_figure(win2['-CANVAS6-'].TKCanvas, distribut_func(sample, distrb))
                win2['-OUT3-'].update('')
                win2['-OUT3-'].print(reproducing_params(sample, distrb))

            if distrb == 'Нормальний':

                if fig_dest is not None:
                    delete_figure_agg(fig_dest)
                fig_dest = draw_figure(win2['-CANVAS5-'].TKCanvas, density_function(sample, distrb))
                if fig_dist is not None:
                    delete_figure_agg(fig_dist)
                fig_dist = draw_figure(win2['-CANVAS6-'].TKCanvas, distribut_func(sample, distrb))
                win2['-OUT3-'].update('')
                win2['-OUT3-'].print(reproducing_params(sample, distrb))

            elif distrb == 'Арксінуса':

                if fig_dest is not None:
                    delete_figure_agg(fig_dest)
                fig_dest = draw_figure(win2['-CANVAS5-'].TKCanvas, density_function(sample, distrb))
                if fig_dist is not None:
                    delete_figure_agg(fig_dist)
                fig_dist = draw_figure(win2['-CANVAS6-'].TKCanvas, distribut_func(sample, distrb))
                win2['-OUT3-'].update('')
                win2['-OUT3-'].print(reproducing_params(sample, distrb))
            elif distrb == 'Вейбула':
                if fig_dest is not None:
                    delete_figure_agg(fig_dest)
                fig_dest = draw_figure(win2['-CANVAS5-'].TKCanvas, density_function(sample, distrb))
                if fig_dist is not None:
                    delete_figure_agg(fig_dist)
                fig_dist = draw_figure(win2['-CANVAS6-'].TKCanvas, distribut_func(sample, distrb))
                win2['-OUT3-'].update('')
                win2['-OUT3-'].print(reproducing_params(sample, distrb))
            elif distrb == 'Рівномірний':
                if fig_dest is not None:
                    delete_figure_agg(fig_dest)
                fig_dest = draw_figure(win2['-CANVAS5-'].TKCanvas, density_function(sample, distrb))
                if fig_dist is not None:
                    delete_figure_agg(fig_dist)
                fig_dist = draw_figure(win2['-CANVAS6-'].TKCanvas, distribut_func(sample, distrb))
                win2['-OUT3-'].update('')
                win2['-OUT3-'].print(reproducing_params(sample, distrb))
        if event2 == sg.WIN_CLOSED or event2 == 'Exit':
            break
    win2.close()


def density_function(sample, distrb):
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.title('Функція щільності')

    n = len(sample)
    avr = average(sample)
    avr_sq = average_sq(sample)

    if distrb == 'Експоненційний':

        lambd = 1 / avr
        y = lambd * np.exp(-lambd * sample)

    elif distrb == 'Арксінуса':

        a = (2 ** 1 / 2) * ((avr_sq - avr ** 2) ** 1 / 2)

        y = 1 / (np.pi * ((a ** 2 - sample ** 2) ** (1 / 2)))

    elif distrb == 'Нормальний':
        m = avr
        sq = (n * ((avr_sq - avr ** 2) ** (1 / 2))) / (n - 1)

        y = (np.exp(-((sample - m) ** 2) / (2 * (sq ** 2)))) / (sq * ((2 * np.pi) ** (1 / 2)))


    elif distrb == 'Вейбула':

        ecdf = ECDF(sample)

        a11 = n - 1

        a12 = 0

        for i in range(n - 1):
            a12 += np.log(sample[i])

        a22 = 0

        for i in range(n - 1):
            a22 += np.log(sample[i]) ** 2

        b1 = 0

        for i in range(1, n - 1):
            # b1 += np.log(np.log(1 / (1 - (i / n))))
            b1 += np.log(np.log(1 / (1 - (ecdf.y[i] / n))))

        b2 = 0

        for i in range(n - 1):
            j = i
            # b2 += np.log(sample[i]) * np.log(np.log(1 / (1 - (i / n))))
            if i == (n - 1):
                b2 += np.log(sample[i]) * np.log(np.log(1 / (1 - (ecdf.y[i] / n))))
            b2 += np.log(sample[i]) * np.log(np.log(1 / (1 - (ecdf.y[j + 1] / n))))

        a_matr = [[a11, a12],

                  [a12, a22]]

        b_matr = [b1, b2]

        a_matr_inv = funcReversMatr(a_matr, 2)

        cof_matr = np.dot(a_matr_inv, b_matr)

        alf = np.exp(-cof_matr[0])

        beta = cof_matr[1]

        y = (beta * (sample ** (beta - 1)) * np.exp(-(sample ** beta) / alf)) / alf



    elif distrb == 'Рівномірний':
        a = avr - ((3 * (avr_sq - avr ** 2)) ** (1 / 2))
        b = avr + ((3 * (avr_sq - avr ** 2)) ** (1 / 2))

        y = [0 for i in range(n)]

        i = 0
        while i < n:
            if sample[i] >= a and sample[i] < b:
                y[i] = 1 / (b - a)
            i += 1

    plt.plot(sample, y)

    # ax.scatter(sample, y)

    return plt.gcf()


def distribut_func(sample, distrb):
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.title('Функція розподілу')

    # if len(sample) < 100:
    #     bins = round((len(v) ** (1 / 2)))
    # else:
    #     bins = round((len(sample) ** (1 / 3)))
    n = len(sample)
    avr = average(sample)
    avr_sq = average_sq(sample)

    conf_inter = 1.36 / n

    # plt.ylim(0, 1)
    if distrb == 'Експоненційний':

        lambd = 1 / avr

        y = 1 - np.exp(-lambd * sample)
        y_up = 1 - np.exp(-lambd * sample) + conf_inter
        y_low = 1 - np.exp(-lambd * sample) - conf_inter

    elif distrb == 'Арксінуса':

        a = (2 ** (1 / 2)) * ((avr_sq - avr ** 2) ** (1 / 2))

        y = 1 / 2 + np.arcsin((sample / a)) / np.pi
        y_up = 1 / 2 + np.arcsin((sample / a)) / np.pi + conf_inter
        y_low = 1 / 2 + np.arcsin((sample / a)) / np.pi - conf_inter


    elif distrb == 'Нормальний':
        m = avr
        sq = (n * ((avr_sq - avr ** 2) ** (1 / 2))) / (n - 1)
        u = (sample - m) / sq
        t = 1 / (1 + 0.2316419 * u)

        # i = 0
        # y = []
        # while i < n:
        #     if u[i] < 0:
        #         y.append(1 - (1 - (np.exp(-(abs(u[i]) ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
        #     -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (10 ** (-8))))
        #     else:
        #         y.append(1 - (np.exp(-(u ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
        #     -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (10 ** (-8)))
        #     i += 1

        y = 1 - (np.exp(-(u ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
            -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (10 ** (-8))

        y_up = 1 - (np.exp(-(u ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
            -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (10 ** (-8)) + conf_inter

        y_low = 1 - (np.exp(-(u ** 2) / 2) * (0.31938153 * t + (-0.356563782) * (t ** 2) + 1.781477937 * (t ** 3) + (
            -1.821255978) * (t ** 4) + 1.330274429 * (t ** 5))) / ((2 * np.pi) ** (1 / 2)) + 7.8 * (
                           10 ** (-8)) - conf_inter


    elif distrb == 'Вейбула':
        ecdf = ECDF(sample)

        a11 = n - 1
        a12 = 0
        for i in range(n - 1):
            a12 += np.log(sample[i])

        a22 = 0
        for i in range(n - 1):
            a22 += np.log(sample[i]) ** 2

        b1 = 0

        for i in range(1, n - 1):
            # b1 += np.log(np.log(1 / (1 - (i / n))))
            b1 += np.log(np.log(1 / (1 - (ecdf.y[i] / n))))

        b2 = 0

        for i in range(n - 1):
            j = i
            # b2 += np.log(sample[i]) * np.log(np.log(1 / (1 - (i / n))))
            if i == (n - 1):
                b2 += np.log(sample[i]) * np.log(np.log(1 / (1 - (ecdf.y[i] / n))))
            b2 += np.log(sample[i]) * np.log(np.log(1 / (1 - (ecdf.y[j + 1] / n))))

        a_matr = [[a11, a12],
                  [a12, a22]]

        b_matr = [b1, b2]

        a_matr_inv = funcReversMatr(a_matr, 2)

        cof_matr = np.dot(a_matr_inv, b_matr)

        alf = np.exp(-cof_matr[0])
        beta = cof_matr[1]

        y = 1 - np.exp(-(sample ** beta) / alf)
        y_up = 1 - np.exp(-(sample ** beta) / alf) + conf_inter
        y_low = 1 - np.exp(-(sample ** beta) / alf) - conf_inter

        # np.log(1 / (1 - (i / len(data))))


    elif distrb == 'Рівномірний':
        a = avr - ((3 * (avr_sq - avr ** 2)) ** (1 / 2))
        b = avr + ((3 * (avr_sq - avr ** 2)) ** (1 / 2))

        y = [0 for i in range(n)]

        i = 0
        while i < n:
            if sample[i] >= a and sample[i] < b:
                y[i] = (sample[i] - a) / (b - a)
            elif sample[i] >= b:
                y[i] = 1
            i += 1

        y_up = [0 for j in range(n)]
        y_low = [0 for j in range(n)]
        for k in range(n):
            y_up[k] = y[k] + conf_inter
            y_low[k] = y[k] - conf_inter



    plt.plot(sample, y)
    plt.plot(sample, y_up)
    plt.plot(sample, y_low)

    # ax.scatter(sample, y)

    return plt.gcf()


def reproducing_params(sample, distrb):
    res = ''
    n = len(sample)
    avr = average(sample)
    avr_sq = average_sq(sample)
    if distrb == 'Експоненційний':
        res += 'Параметри Експоненційного розподілу: \n'

        lambd = 1 / avr
        res += '\u03BB: ' + str(round(lambd, 4)) + '\n'

        res += 'E(\u03BE): ' + str(round(avr, 4)) + '\n'

        disp = 1 / (lambd ** 2)
        res += 'D(\u03BE): ' + str(round(disp, 4)) + '\n'

        res += 'A: 2\n'
        res += 'E: 6\n'

        lambd_disp = (lambd ** 2) / n
        res += 'D(\u03BB): ' + str(round(lambd_disp, 4)) + '\n'

        return res


    elif distrb == 'Нормальний':
        res += 'Параметри Нормального розподілу: \n'
        m = avr
        sq = (n * ((avr_sq - avr ** 2) ** (1 / 2))) / (n - 1)
        res += 'm: ' + str(round(m, 4)) + '\n'
        res += '\u03C3: ' + str(round(sq, 4)) + '\n'

        res += 'E(\u03BE): ' + str(round(m, 4)) + '\n'

        res += 'D(\u03BE): ' + str(round(sq ** 2, 4)) + '\n'

        res += 'A: 0\n'
        res += 'E: 0\n'
        res += 'E\u0302: 3\n'

        disp_m = (sq ** 2) / n
        res += 'D(m\u0302:)' + str(round(disp_m, 4)) + '\n'

        disp_sq = sq ** 2 / (2 * n)
        res += 'D(\u03C3\u0302 ): ' + str(round(disp_sq, 4)) + '\n'

        res += 'cov(m\u0302, \u03C3\u0302 ): 0\n'

        return res



    elif distrb == 'Арксінуса':
        res += 'Параметри розподілу Арксінуса: \n'
        a = (2 ** 1 / 2) * ((avr_sq - avr ** 2) ** 1 / 2)
        res += 'a: ' + str(round(a, 4)) + '\n'

        res += 'E(\u03BE): 0\n'

        disp = (a ** 2) / 2
        res += 'D(\u03BE): ' + str(round(disp, 4)) + '\n'

        res += 'A: 0\n'
        res += 'E: -1.5\n'

        disp_a = (a ** 4) / (8 * n)
        res += 'D(a): ' + str(round(disp_a, 4)) + '\n'

        return res

    elif distrb == 'Вейбула':
        res += 'Параметри розподілу Вейбула: \n'



    elif distrb == 'Рівномірний':
        res += 'Параметри Рівномірного розподілу: \n'

        a = avr - ((3 * (avr_sq - avr ** 2)) ** (1 / 2))
        b = avr + ((3 * (avr_sq - avr ** 2)) ** (1 / 2))

        res += 'a: ' + str(round(a, 4)) + '\n'
        res += 'b: ' + str(round(b, 4)) + '\n'

        disp_avr = ((b - a) ** 2) / (12 * n)
        res += 'D(x\u0302): ' + str(round(disp_avr, 4)) + '\n'

        disp_avr_sq = (((b - a) ** 4) + 15 * ((a + b) ** 2) * ((b - a) ** 2)) / (180 * n)
        res += 'D(x^2\u0302): ' + str(round(disp_avr_sq, 4)) + '\n'

        cov = ((a + b) * ((b - a) ** 2)) / (12 * n)
        res += 'cov(x\u0302, x^2\u0302): ' + str(round(cov, 4)) + '\n'

        return res



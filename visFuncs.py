import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
import numpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import plotly.express as px
import seaborn as sns
from scipy import stats
from funcs import *
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import arcsine
from paramFuncs import *


def create_histogram(v, classes=None):
    fig, ax = plt.subplots(figsize=(5, 4))

    if classes:
        b = classes
    else:
        if len(v) < 100:
            b = round((len(v) ** (1 / 2)))
        else:
            b = round((len(v) ** (1 / 3)))

    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.xlabel('Варіанти')
    plt.ylabel('Частоти')

    plt.title('Відносні частоти')

    plt.hist(v, bins=b, edgecolor="black", color='blue', weights=np.ones_like(v) / len(v))

    return plt.gcf()


def create_distribution_function(data, classes=None):
    fig, ax = plt.subplots(figsize=(5, 4))

    plt.grid(color='grey', linestyle='--', linewidth=0.5)

    n = len(data)

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
    ax.scatter(x=data, y=s_y, s=5)
    sns.histplot(data, element="step", fill=False,
                 cumulative=True, stat="density", common_norm=False, bins=b, color='red')

    plt.xlim(data[0], data[-1])
    # plt.ylim(0, 1)

    """thoughts about distribution function"""
    # avr = average(data)
    # sq = averageSq(data, avr)
    #
    # x = np.sort(data)
    # y1 = (np.arange(len(x)) / float(len(x)))
    # plt.plot(x, y1, color='red')
    #
    # y2 = ((np.arange(len(x)) / float(len(x)))) + t * sq
    # plt.plot(x, y2, color='green', linestyle='--')
    #
    # y3 = ((np.arange(len(x)) / float(len(x)))) - t * sq
    # plt.plot(x, y3, color='black', linestyle='--')

    # y_vals1 = np.array([exp_pdf(avr, x) for x in data])
    #
    # ax.plot(data, y_vals1)

    # cdf = norm_rv.cdf()

    # ecdf = ECDF(data)

    # sns.lineplot(x=data, y=s_y, color='g')
    # sns.lineplot(x=data, y=s_y + t * sq, color='r')
    # sns.lineplot(x=data, y=s_y - t * sq, color='r')

    # plt.step(ecdf.x, ecdf.y, 'g')

    # sns.lineplot(x=data, y=s_y, color='green')
    # sns.kdeplot(data=data, cumulative=True, color='green')

    plt.xlabel('')
    plt.ylabel('')

    plt.title('Функція розподілу')

    return plt.gcf()


# def exp_pdf(lam, x):
#     return 1 - np.exp(-lam * x)


def create_probability_grid(data):
    fig, ax = plt.subplots(figsize=(5, 4))

    y_p = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    y_ax = []

    for i in range(len(y_p)):
        y = np.log(1 / (1 - y_p[i]))
        y_ax.append(y)

    plt.ylim(0, y_ax[-1])

    for i in range(len(y_ax)):
        ax.axhline(y_ax[i], linewidth=1, color='r')
        plt.yticks(y_ax)
    ax.set_yticklabels(y_p, fontdict={'fontsize': 7})

    for i in range(len(data)):
        plt.plot(data[i], np.log(1 / (1 - (i / len(data)))), marker="o", markersize=2.5)

    plt.xlabel('')
    plt.ylabel('F(x)')
    plt.grid(axis='x')

    plt.title('Імовірнісна сітка')

    return plt.gcf()


def density_function(sample, distrb):
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.title('Функція щільності')

    avr = average(sample)
    avr_sq = average_sq(sample)

    if distrb == 'Експоненційний':

        lambd = 1 / avr
        y = lambd * np.exp(-lambd * sample)

    elif distrb == 'Арксінуса':


        a = (2 ** 1 / 2) * ((avr_sq - avr**2) ** 1 / 2)

        y = 1 / (np.pi * ((a ** 2 - sample ** 2) ** (1 / 2)))

    # elif distrb == 'Нормальний':
    #     m = avr
    #     sq = len(sample) * ((avr_sq - avr**2))



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
    avr = average(sample)
    avr_sq = average_sq(sample)

    if distrb == 'Експоненційний':

        lambd = 1 / avr

        y = 1 - np.exp(-lambd * sample)

    elif distrb == 'Арксінуса':


        a = (2 ** (1 / 2)) * ((avr_sq - avr**2) ** (1 / 2))

        y = 1 / 2 + np.arcsin((sample / a)) / np.pi

    # elif distrb == 'Нормальний':




    plt.plot(sample, y)
    # ax.scatter(sample, y)

    return plt.gcf()


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


def reproducing_distributions():
    menu_def2 = [['Меню', ['Відкрити файл']]]
    layout2 = [[sg.Menu(menu_def2)],
               [sg.Text('Розподіл'), sg.Combo(['Експоненційний', 'Нормальний', 'Рівномірний', 'Вейбула', 'Арксінуса'],
                                              font=('Arial Bold', 14), enable_events=True, readonly=False,
                                              key='-COMBO-'), sg.Button('Відтворити')],
               [sg.Canvas(size=(4, 3), key='-CANVAS5-'), sg.Push(), sg.Canvas(size=(4, 3), key='-CANVAS6-')]]

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

            if distrb == 'Нормальний':

                if fig_dest is not None:
                    delete_figure_agg(fig_dest)
                fig_dest = draw_figure(win2['-CANVAS5-'].TKCanvas, density_function(sample))
                if fig_dist is not None:
                    delete_figure_agg(fig_dist)
                fig_dist = draw_figure(win2['-CANVAS6-'].TKCanvas, distribut_func(sample))


            elif distrb == 'Рівномірний':

                if fig_dest is not None:
                    delete_figure_agg(fig_dest)
                fig_dest = draw_figure(win2['-CANVAS5-'].TKCanvas, density_function(sample))
                if fig_dist is not None:
                    delete_figure_agg(fig_dist)
                fig_dist = draw_figure(win2['-CANVAS6-'].TKCanvas, distribut_func(sample))



            elif distrb == 'Арксінуса':
                # sample = arcsine.rvs(size=1000)
                # sample = shellSort(sample, len(sample))

                if fig_dest is not None:
                    delete_figure_agg(fig_dest)
                fig_dest = draw_figure(win2['-CANVAS5-'].TKCanvas, density_function(sample, distrb))
                if fig_dist is not None:
                    delete_figure_agg(fig_dist)
                fig_dist = draw_figure(win2['-CANVAS6-'].TKCanvas, distribut_func(sample, distrb))

        # elif distrb == 'Вейбула':
        #
        # elif distrb == 'Вейбула':
        #
        # elif distrb == 'Арксінуса':

        if event2 == sg.WIN_CLOSED or event2 == 'Exit':
            break
    win2.close()

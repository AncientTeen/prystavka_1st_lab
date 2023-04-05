import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import plotly.express as px
import seaborn as sns
from scipy import stats
from funcs import *
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

    if classes:
        b = classes
    else:
        if len(data) < 100:
            b = round((len(data) ** (1 / 2)))
        else:
            b = round((len(data) ** (1 / 3)))

    s_y = np.arange(1, len(data) + 1) / len(data)
    ax.scatter(x=data, y=s_y, s=7)
    sns.histplot(data, element="step", fill=False,
                 cumulative=True, stat="density", common_norm=False, bins=b, color='red')

    plt.xlabel('')
    plt.ylabel('')

    plt.title('Функція розподілу')

    return plt.gcf()




def create_probability_grid(data):
    fig, ax = plt.subplots(figsize=(5, 4))

    n = len(data)
    y_ax = []
    for i in range(n):
        y = np.log(1 / (1 - round(np.quantile(data[i], 0.05), 3)))
        y_ax.append(y)

    ax.scatter(x=data, y=y_ax, s=7)

    plt.xlabel('')
    plt.ylabel('')

    plt.title('Імовірнісна сітка')

    return plt.gcf()

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')

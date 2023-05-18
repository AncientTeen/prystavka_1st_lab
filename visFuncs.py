from paramFuncs import *


def create_histogram(v, classes=None):
    fig, ax = plt.subplots(figsize=(5, 4))

    if classes:
        b = classes
    else:
        if len(v) < 100:
            b = round((len(v) ** (1 / 2)))
            if b % 2 == 0:
                b -= 1
        else:
            b = round((len(v) ** (1 / 3)))
            if b % 2 == 0:
                b -= 1

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

    plt.xlabel('')
    plt.ylabel('')

    plt.title('Функція розподілу')

    return plt.gcf()


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


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')

from paramFuncs import *
from visFuncs import *

menu_def = [['Меню', ['Відкрити файл', 'Точкові оцінки', 'Перетворення',
                      ['Логарифмувати', 'Стандартизувати', 'Вилучення аномальних значень'], 'Вийти']]]
sg.theme('DarkTeal1')
layout = [[sg.Menu(menu_def)],
          [sg.Button('Гістограма'), sg.Button('Стерти'), sg.Push(), sg.Button('Функція розподілу')],
          [sg.Text("Кількість класів"), sg.InputText(size=(15, 1), key='-IN1-'),
           sg.Button('Ок')],
          [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.Canvas(size=(4, 3), key='-CANVAS2-')],
          # [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.Output(size=(100, 100), key='-Output-')],
          [sg.VPush()],
          [sg.VPush()],
          [sg.VPush()],
          [sg.VPush(),
           sg.Multiline(size=(50, 10), key='-OUT1-', reroute_stdout=True, do_not_clear=True, font='Comic, 15'),
           sg.Multiline(size=(50, 10), key='-OUT2-', do_not_clear=True, font='Comic, 15')],
          ]

"""thoughts about tabs"""
# l1 = [sg.Multiline(size=(50, 10), key='-OUT1-', reroute_stdout=True, do_not_clear=False, font='Comic, 15')]
# l2 = [sg.Multiline(size=(50, 10), key='-OUT2-', do_not_clear=False, font='Comic, 15')]
#
# menu_def = [['Меню', ['Відкрити файл', 'Точкові оцінки', 'Логарифмувати', 'Стандартизувати', 'Вийти']]]
# sg.theme('DarkTeal1')
# layout = [[sg.Menu(menu_def)],
#           [sg.Button('Гістограма'), sg.Button('Стерти'), sg.Push(), sg.Button('Функція розподілу')],
#           [sg.Text("Кількість класів"), sg.InputText(size=(15, 1), key='-IN1-'),
#            sg.Button('Ок')],
#           [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.Canvas(size=(4, 3), key='-CANVAS2-')],
#           # [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.Output(size=(100, 100), key='-Output-')],
#           [sg.VPush()],
#           [sg.VPush()],
#           [sg.VPush()],
#           [sg.VPush()],
#           [sg.TabGroup([[sg.Tab('Comb', layout1), sg.Tab('Newton', layout2)]]],

window = sg.Window('1st lab_work Prystavka', layout, size=(1200, 800))

fig_hist = None
fig_ecdf = None
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Вийти'):
        break

    if event == 'Відкрити файл':
        filename = sg.popup_get_file('file to open', no_window=True)
        nums = []
        with open(filename) as d:
            num = d.readline()
            while num:
                nums.append(float(num))
                num = d.readline()
        d.close()
        nums = shellSort(nums, len(nums))
        create_histogram(nums)
        create_distribution_function(nums)

    if event == 'Точкові оцінки':
        window['-OUT1-'].update('')
        window['-OUT2-'].update('')
        window['-OUT1-'].print(paramFunc(nums))
        window['-OUT2-'].print(nums)

    if event == 'Логарифмувати':
        nums = logs(nums)

    if event == 'Стандартизувати':
        avr = average(nums)
        avrSq = averageSq(nums, avr)
        nums = standr(nums, avr, avrSq)

    if event == 'Вилучення аномальних значень':
        nums = removeAnomalian(nums)

    if event == 'Гістограма':
        if fig_hist is not None:
            delete_figure_agg(fig_hist)
        fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))
    if event == 'Стерти':
        if fig_hist and fig_ecdf is not None:
            delete_figure_agg(fig_hist)
            delete_figure_agg(fig_ecdf)

    if event == 'Ок':
        if fig_hist is not None:
            delete_figure_agg(fig_hist)
        if fig_ecdf is not None:
            delete_figure_agg(fig_ecdf)
        fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums, int(values['-IN1-'])))
        fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums, int(values['-IN1-'])))

    if event == 'Функція розподілу':
        if fig_ecdf is not None:
            delete_figure_agg(fig_ecdf)
        fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))
        # if fig_ecdf is not None:
        #     delete_figure_agg(fig_ecdf)
        # window['-Output-'].print(create_distribution_function(nums))

window.close()

# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import PySimpleGUI as sg
# import matplotlib, time, threading
#
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def fig_maker(window):  # this should be called as a thread, then time.sleep() here would not freeze the GUI
#     plt.scatter(np.random.rand(1, 10), np.random.rand(1, 10))
#     window.write_event_value('-THREAD-', 'done.')
#     time.sleep(1)
#     return plt.gcf()
#
#
# def draw_figure(canvas, figure, loc=(0, 0)):
#     figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
#     figure_canvas_agg.draw()
#     figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
#     return figure_canvas_agg
#
#
# def delete_fig_agg(fig_agg):
#     fig_agg.get_tk_widget().forget()
#     plt.close('all')
#
#
# if __name__ == '__main__':
#     # define the window layout
#     layout = [[sg.Button('update'), sg.Button('Stop', key="-STOP-"), sg.Button('Exit', key="-EXIT-")],
#               [sg.Radio('Keep looping', "RADIO1", default=True, size=(12, 3), key="-LOOP-"),
#                sg.Radio('Stop looping', "RADIO1", size=(12, 3), key='-NOLOOP-')],
#               [sg.Text('Plot test', font='Any 18')],
#               [sg.Canvas(size=(500, 500), key='canvas')]]
#
#     # create the form and show it without the plot
#     window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI',
#                        layout, finalize=True)
#
#     fig_agg = None
#     while True:
#         event, values = window.read()
#         if event is None:  # if user closes window
#             break
#
#         if event == "update":
#             if fig_agg is not None:
#                 delete_fig_agg(fig_agg)
#             fig = fig_maker(window)
#             fig_agg = draw_figure(window['canvas'].TKCanvas, fig)
#
#         if event == "-THREAD-":
#             print('Acquisition: ', values[event])
#             time.sleep(1)
#             if values['-LOOP-'] == True:
#                 if fig_agg is not None:
#                     delete_fig_agg(fig_agg)
#                 fig = fig_maker(window)
#                 fig_agg = draw_figure(window['canvas'].TKCanvas, fig)
#                 window.Refresh()
#
#         if event == "-STOP-":
#             window['-NOLOOP-'].update(True)
#
#         if event == "-EXIT-":
#             break
#
#     window.close()
# import numpy as np
# # mu, sigma = 5, 1
# # rand_normal = np.random.normal(mu, sigma, 100)
# n = 10
# y = np.arange(1, n+1) / n
#
# print(y)

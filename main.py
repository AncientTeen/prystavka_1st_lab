import numpy as np

from paramFuncs import *
from visFuncs import *

# menu_def = [['Меню', ['Відкрити файл', 'Точкові оцінки', 'Перетворення',
#                       ['Логарифмувати', 'Стандартизувати', 'Вилучення аномальних значень', ], 'Вийти']]]
sg.theme('DarkBlue7')
# layout = [[sg.Menu(menu_def)],
#           [sg.Button('Гістограма'), sg.Button('Стерти'), sg.Push(), sg.Button('Функція розподілу')],
#           [sg.Text("Кількість класів"), sg.InputText(size=(15, 1), key='-IN1-'),
#            sg.Button('Ок')],
#           [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.Canvas(size=(4, 3), key='-CANVAS2-')],
#           # [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.Output(size=(100, 100), key='-Output-')],
#           [sg.VPush()],
#           [sg.VPush()],
#           [sg.VPush()],
#           [sg.VPush(),
#            sg.Multiline(size=(50, 10), key='-OUT1-', reroute_stdout=True, do_not_clear=True, font='Comic, 15'),
#            sg.Multiline(size=(50, 10), key='-OUT2-', do_not_clear=True, font='Comic, 15')],
#           ]

"""thoughts about tabs"""
l1 = [[sg.Multiline(size=(100, 10), key='-OUT1-', reroute_stdout=True, do_not_clear=True, font='Comic, 15')]]
l2 = [[sg.Multiline(size=(100, 10), key='-OUT2-', do_not_clear=True, font='Comic, 15')]]

w1 = [[sg.Canvas(size=(4, 3), key='-CANVAS2-')]]
w2 = [[sg.Canvas(size=(4, 3), key='-CANVAS3-')]]

menu_def = [['Меню', ['Відкрити файл', 'Перетворення',
                      ['Логарифмувати', 'Стандартизувати', 'Вилучення аномальних значень', ],
                      'Відтворення розподілів', 'Стерти', 'Вийти']]]
layout = [[sg.Menu(menu_def)],
          [sg.Text("Кількість класів"), sg.InputText(size=(15, 1), key='-IN1-'),
           sg.Button('Ок')],
          # [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.Canvas(size=(4, 3), key='-CANVAS2-')],
          [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.TabGroup([
              [sg.Tab('Функція розподілу', w1),
               sg.Tab('Імовірністна сітка', w2)]])],

          # [sg.Canvas(size=(4, 3), key='-CANVAS1-'), sg.Push(), sg.Output(size=(100, 100), key='-Output-')],
          [sg.VPush()],
          [sg.VPush()],
          [sg.VPush()],
          [sg.VPush()],
          [sg.TabGroup([
              [sg.Tab('Протокол', l1),
               sg.Tab('Об\'єкти', l2)]])],
          ]

window = sg.Window('1st lab_work Prystavka', layout, size=(1200, 800))

fig_hist = None
fig_ecdf = None
fig_grid = None


while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Вийти'):
        break

    if event == 'Відкрити файл':
        filename = sg.popup_get_file('file to open', no_window=True)
        # nums = []
        # with open(filename) as d:
        #     num = d.readline()
        #     while num:
        #         nums.append(float(num))
        #         num = d.readline()
        # d.close()

        nums = []
        with open(filename) as d:
            num = d.readline()
            while num:
                if len(num) == 1:
                    nums.append(float(num))
                else:
                    s_nums = num.split()
                    for i in range(len(s_nums)):
                        nums.append(float(s_nums[i]))

                num = d.readline()
        d.close()


        # nums = np.fromfile(filename, dtype=float)

        nums = shellSort(nums, len(nums))
        create_histogram(nums)
        create_distribution_function(nums)

        if fig_hist is not None:
            delete_figure_agg(fig_hist)
        fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))

        if fig_ecdf is not None:
            delete_figure_agg(fig_ecdf)
        fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))

        if fig_grid is not None:
            delete_figure_agg(fig_grid)
        fig_grid = draw_figure(window['-CANVAS3-'].TKCanvas, create_probability_grid(nums))

        window['-OUT1-'].update('')
        window['-OUT2-'].update('')
        window['-OUT1-'].print(paramFunc(nums))
        window['-OUT2-'].print(nums)

    if event == 'Логарифмувати':
        nums = logs(nums)

        if fig_hist is not None:
            delete_figure_agg(fig_hist)
        fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))

        if fig_ecdf is not None:
            delete_figure_agg(fig_ecdf)
        fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))

        if fig_grid is not None:
            delete_figure_agg(fig_grid)
        fig_grid = draw_figure(window['-CANVAS3-'].TKCanvas, create_probability_grid(nums))

        window['-OUT1-'].update('')
        window['-OUT2-'].update('')
        window['-OUT1-'].print(paramFunc(nums))
        window['-OUT2-'].print(nums)

    if event == 'Стандартизувати':

        nums = standr(nums)

        if fig_hist is not None:
            delete_figure_agg(fig_hist)
        fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))

        if fig_ecdf is not None:
            delete_figure_agg(fig_ecdf)
        fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))

        if fig_grid is not None:
            delete_figure_agg(fig_grid)
        fig_grid = draw_figure(window['-CANVAS3-'].TKCanvas, create_probability_grid(nums))

        window['-OUT1-'].update('')
        window['-OUT2-'].update('')
        window['-OUT1-'].print(paramFunc(nums))
        window['-OUT2-'].print(nums)

    if event == 'Вилучення аномальних значень':
        nums = removeAnomalous(nums)

        if fig_hist is not None:
            delete_figure_agg(fig_hist)
        fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums))

        if fig_ecdf is not None:
            delete_figure_agg(fig_ecdf)
        fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums))

        if fig_grid is not None:
            delete_figure_agg(fig_grid)
        fig_grid = draw_figure(window['-CANVAS3-'].TKCanvas, create_probability_grid(nums))

        window['-OUT1-'].update('')
        window['-OUT2-'].update('')
        window['-OUT1-'].print(paramFunc(nums))
        window['-OUT2-'].print(nums)

    if event == 'Стерти':
        if fig_hist and fig_ecdf is not None:
            delete_figure_agg(fig_hist)
            delete_figure_agg(fig_ecdf)
        window['-OUT1-'].update('')
        window['-OUT2-'].update('')

    if event == 'Ок':
        if fig_hist is not None:
            delete_figure_agg(fig_hist)
        if fig_ecdf is not None:
            delete_figure_agg(fig_ecdf)
        fig_hist = draw_figure(window['-CANVAS1-'].TKCanvas, create_histogram(nums, int(values['-IN1-'])))
        fig_ecdf = draw_figure(window['-CANVAS2-'].TKCanvas, create_distribution_function(nums, int(values['-IN1-'])))

    if event == 'Відтворення розподілів':
        reproducing_distributions()


window.close()

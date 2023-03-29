import numpy as np
from funcs import *


def average(data):
    avr = 0
    for i in range(len(data)):
        avr += data[i]
    avr = round((avr / len(data)), 3)
    return avr


def truncatedAverage(data):
    alf = 0.3
    k = round(len(data) * alf)

    trncAvr = 0
    for i in range(k, len(data) - k):
        trncAvr += data[i]
    trncAvr = trncAvr / (len(data) - 2 * k)
    return trncAvr


def medium(data):
    if len(data) % 2 == 0:
        md = round(((data[int(len(data) / 2)] + data[int((len(data) / 2) - 1)]) / 2), 3)

    else:
        md = round((data[len(data) // 2]), 3)
    return md


def mediumWalsh(data):
    sumData = []

    for i in range(len(data) - 1):
        for j in range(len(data) - 1):
            x = (data[i] + data[j + 1]) / 2
            sumData.append(x)

    sumData = shellSort(sumData, len(sumData))
    mdWlsh = medium(sumData)
    return mdWlsh


def mediumAbsMiss(data, md):
    n = len(data)
    arr = []
    for i in range(n):
        x = abs(data[i] - md)
        arr.append(x)

    arr = shellSort(arr, len(arr))
    mdAbsMss = round(1.483 * medium(arr), 3)

    return mdAbsMss


def averageSq(data, avr):
    avrSq = 0

    for i in range(len(data)):
        avrSq += (data[i] - avr) ** 2
    avrSq = round((avrSq / (len(data) - 1)) ** (1 / 2), 3)

    return avrSq


def assymCoef(data, avr):
    shftSq = 0

    for i in range(len(data)):
        shftSq += data[i] ** 2 - avr ** 2
    shftSq = round((shftSq / (len(data))) ** (1 / 2), 3)

    sftAssmCf = 0

    for i in range(len(data)):
        sftAssmCf += (data[i] - avr) ** 3

    sftAssmCf = sftAssmCf / (len(data) * (shftSq ** 3))

    assmCf = round(((((len(data) * (len(data) - 1)) ** (1 / 2)) * sftAssmCf) / (len(data) - 2)), 3)

    return assmCf


def excessCoef(data, avr):
    shftSq = 0

    for i in range(len(data)):
        shftSq += data[i] ** 2 - avr ** 2
    shftSq = round((shftSq / (len(data))) ** (1 / 2), 3)

    shftExCf = 0
    for i in range(len(data)):
        shftExCf += (data[i] - avr) ** 4
    shftExCf = shftExCf / (len(data) * (shftSq ** 4))

    exCf = round(
        (((len(data) ** 2 - 1) / ((len(data) - 2) * (len(data) - 3))) * ((shftExCf - 3) + (6 / (len(data) + 1)))), 3)

    return exCf


def contrExcessCoef(exCf):
    cntrExCf = round((1 / ((abs(exCf)) ** (1 / 2))), 3)
    return cntrExCf


def pirsonCoef(avrSq, avr):
    if avr == 0:
        return None

    prsCf = round((avrSq / avr), 3)
    return prsCf


def nonParamCoefVar(mdAbsMss, md):
    return round(mdAbsMss / md, 3)


def confInterAvr(data):
    freed = len(data)
    t = 0
    if freed < 120:
        if freed == 69:
            t = 1.995
        elif freed == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    sq = averageSq(data, avr)

    inf = round(avr - t * sq / (len(data) ** (1 / 2)), 4)
    sup = round(avr + t * sq / (len(data) ** (1 / 2)), 4)

    return inf, sup


def confInterSqAvr(data):
    n = len(data)

    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    sq = averageSq(data, avr)

    inf = round(sq - t * sq * (2 / (n - 1)) ** (1 / 4), 4)
    sup = round(sq + t * sq * (2 / (n - 1)) ** (1 / 4), 4)

    return inf, sup


def confInterAssym(data):
    n = len(data)
    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    assymCof = assymCoef(data, avr)

    inf = round(assymCof - t * (6 * (n - 2) / ((n + 1) * (n + 3))) ** (1 / 2), 4)
    sup = round(assymCof + t * (6 * (n - 2) / ((n + 1) * (n + 3))) ** (1 / 2), 4)

    return inf, sup


def confInterExcess(data):
    n = len(data)
    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    exCf = excessCoef(data, avr)

    inf = round(exCf - t * (24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) ** (1 / 2), 4)
    sup = round(exCf + t * (24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) ** (1 / 2), 4)

    return inf, sup


def confInterContrEx(data):
    n = len(data)
    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)

    shftSq = 0

    for i in range(n):
        shftSq += data[i] ** 2 - avr ** 2
    shftSq = round((shftSq / n) ** (1 / 2), 3)

    shftExCf = 0
    for i in range(n):
        shftExCf += (data[i] - avr) ** 4
    shftExCf = shftExCf / (n * (shftSq ** 4))

    exCf = excessCoef(data, avr)
    cntrExCf = contrExcessCoef(exCf)

    # inf = round(cntrExCf - t * (((abs(shftExCf) / (29 * n)) ** (1 / 2)) * (abs((shftExCf ** 2) - 1) ** (3 / 4))),
    #             4)
    # sup = round(cntrExCf + t * (((abs(shftExCf) / (29 * n)) ** (1 / 2)) * (abs((shftExCf ** 2) - 1) ** (3 / 4))),
    #             4)

    # inf = round(cntrExCf - t * ((abs(shftExCf) / (29 * n)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4))) ** (1 / 2), 4)
    # sup = round(cntrExCf + t * ((abs(shftExCf) / (29 * n)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4))) ** (1 / 2), 4)

    inf = round(cntrExCf - t * ((abs(shftExCf) / (29 * n)) ** (1 / 2)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4)), 4)
    sup = round(cntrExCf + t * ((abs(shftExCf) / (29 * n)) ** (1 / 2)) * ((abs(shftExCf ** 2 - 1)) ** (3 / 4)), 4)

    return inf, sup


def confInterVariation(data):
    n = len(data)
    t = 0
    if n - 1 < 120:
        if n - 1 == 69:
            t = 1.995
        elif n - 1 == 24:
            t = 2.06
    else:
        t = 1.96

    avr = average(data)
    avrSq = averageSq(data, avr)
    prsCf = pirsonCoef(avrSq, avr)

    inf = round(prsCf - t * prsCf * (((1 + 2 * prsCf) / (2 * n)) ** (1 / 2)), 4)
    sup = round(prsCf + t * prsCf * (((1 + 2 * prsCf) / (2 * n)) ** (1 / 2)), 4)

    return inf, sup


def logs(data):
    logData = []

    if data[0] < 0:
        for i in range(len(data)):
            x = round((np.log10(data[i] + abs(data[0]) + 0.01)), 3)
            logData.append(x)
    else:
        for i in range(len(data)):
            x = round(np.log10(data[i]), 3)
            logData.append(x)
    return logData


def standr(data, avr, avrSq):
    stnData = []

    for i in range(len(data)):
        x = round(((data[i] - avr) / avrSq), 3)
        stnData.append(x)
    return stnData


def removeAnomalian(data):
    n = len(data)

    arr = []
    avr = average(data)
    avrSq = averageSq(data, avr)
    exCf = excessCoef(data, avr)
    cntrExCf = contrExcessCoef(exCf)

    t = 1.2 + 3.6 * (1 - cntrExCf) * np.log10(n / 10)

    a = avr - t * avrSq
    b = avr + t * avrSq

    for i in range(n):
        if data[i] < a or data[i] > b:
            continue
        arr.append(data[i])

    return arr


def paramFunc(data):
    avr = average(data)
    print('Середнє значення: ', avr)

    trcnAvr = truncatedAverage(data)
    print('Усічене середнє: ', trcnAvr)

    md = medium(data)
    print('Медіана: ', md)

    mdWlsh = mediumWalsh(data)
    print('Медіана Уолша: ', mdWlsh)

    mdAbsMss = mediumAbsMiss(data, md)
    print('Медіана абсолютних відхилень: ', mdAbsMss)

    avrSq = averageSq(data, avr)
    print('Сер. квадратичне: ', avrSq)

    assmCf = assymCoef(data, avr)
    print('Коефіцієнт асиметрії: ', assmCf)

    exCf = excessCoef(data, avr)
    print('Коефіцієнт ексцесу: ', exCf)

    cntrExCf = contrExcessCoef(exCf)
    print('Коефіцієнт контрексцесу: ', cntrExCf)

    prsCf = pirsonCoef(avrSq, avr)
    print('Коефіцієнт Пірсона: ', prsCf)

    nonParamCfVar = nonParamCoefVar(mdAbsMss, md)
    print('Непараметричний коефіцієнт варіації: ', nonParamCfVar)

    print('Квантилі : ')
    print('0.05: ', round(np.quantile(data, 0.05), 3))
    print('0.1: ', round(np.quantile(data, 0.1), 3))
    print('0.25: ', round(np.quantile(data, 0.25), 3))
    print('0.5: ', round(np.quantile(data, 0.5), 3))
    print('0.75: ', round(np.quantile(data, 0.75), 3))
    print('0.9: ', round(np.quantile(data, 0.9), 3))
    print('0.95: ', round(np.quantile(data, 0.95), 3))

    avrIntr = confInterAvr(data)
    print('inf середнього: ', avrIntr[0])
    print('sup середнього: ', avrIntr[1])

    sqIntr = confInterSqAvr(data)
    print('inf середнього квадратичного: ', sqIntr[0])
    print('sup середнього квадратичного: ', sqIntr[1])

    assmIntrCof = confInterAssym(data)
    print('inf асиметрії: ', assmIntrCof[0])
    print('sup асиметрії: ', assmIntrCof[1])

    exIntrCof = confInterExcess(data)
    print('inf ексцесу: ', exIntrCof[0])
    print('sup ексцесу: ', exIntrCof[1])

    cExIntrCof = confInterContrEx(data)
    print('inf контрексцесу: ', cExIntrCof[0])
    print('sup контрексцесу: ', cExIntrCof[1])

    variatIntrCof = confInterVariation(data)
    print('inf варіації : ', variatIntrCof[0])
    print('sup варіації: ', variatIntrCof[1])

    # anom = removeAnomalian(data)
    # print('a ', anom[0])
    # print('b ', anom[1])

    return ''

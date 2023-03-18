import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
plt.style.use('seaborn-whitegrid')
import matplotlib.pylab as plt
import csv
from scipy.optimize import curve_fit

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

style = ['-', ':', '--', '-.', '-', ':']
mark = ['.', '*', '+', '1', '2', '3', '4']

if __name__ == '__main__':
    absolute_path = '../result/alexnet/'
    file_name_list = ['result_50.csv', 'result_40.csv', 'result_30.csv', 'result_20.csv', 'result_10.csv', 'result_1.csv']

    i = 0
    for item in file_name_list:
        f = open(absolute_path+item, 'r')
        csv.field_size_limit(1000 * 1024 * 1024)
        f_csv = csv.reader(f)
        qp = []
        compression_rate = []
        for row in f_csv:
            qp.append(int(row[0]))
            compression_rate.append(float(row[-1]))
        print(qp)
        print(compression_rate)
        title = item.split('.')[0]
        title = title.split('_')[-1]
        plt.plot(qp, compression_rate, label='Number of image frames: {}'.format(title), linewidth=2, linestyle=style[i], marker=mark[i])
        i += 1

    plt.xlabel('qp', font)
    plt.ylabel('compression rate', font)
    plt.legend()
    plt.savefig('alexnet_compression_rate.pdf')

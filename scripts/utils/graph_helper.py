#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Graph Helper Functions
# By Nick Erickson

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def graph_simple(x_list, y_list, names_list, title, y_label, x_label, savefig_name=""):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(y_list))))
    plt.figure()
    max_y = 0
    max_x = 0
    min_y = 99999999
    min_x = 99999999
    for i in range(len(y_list)):
        color = next(colors)
        name = names_list[i]
        y = y_list[i]
        x = x_list[i]

        max_y = max(max_y, np.max(y))
        min_y = min(min_y, np.min(y))
        max_x = max(max_x, np.max(x))
        min_x = min(min_x, np.min(x))

        plt.plot(x, y, c=color, label=name)

    plt.legend(loc=2)
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.grid(True)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    if savefig_name != "":
        plt.savefig(savefig_name)
    plt.close()

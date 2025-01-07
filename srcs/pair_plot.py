# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2025-01-07 20:42:50

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        print("Usage: pair_plot.py dataset.csv")
        sys.exit()
    
    data = pd.read_csv(sys.argv[1])
    data.replace('', np.nan, inplace=True)
    data.drop('Index', axis=1, inplace=True)
    house_colors = ['#4682B4', '#2E8B57', '#B22222', '#FFD700']
    
    if not os.path.exists('results/results_pairplot'):
        os.makedirs('results/results_pairplot')
    
    sns.set_palette(sns.color_palette(house_colors))
    pair_plot = sns.pairplot(data, hue='Hogwarts House', diag_kind='hist')
    pair_plot.savefig('results/results_pairplot/pair_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
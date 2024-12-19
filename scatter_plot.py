# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2024-12-19 16:29:22

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        print("Usage: scatter_plot.py dataset.csv")
        sys.exit()
    
    data = pd.read_csv(sys.argv[1])
    data.replace('', np.nan, inplace=True)
    data.drop('Index', axis=1, inplace=True)
    numeric_columns = data.select_dtypes(include=np.number).columns
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results_scatter'):
        os.makedirs('results_scatter')
    
    house_colors = ['#4682B4', '#2E8B57', '#B22222', '#FFD700']
    
    # Generate scatter plots for each pair of numeric columns
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            plt.figure()
            sns.set_palette(sns.color_palette(house_colors))
            sns.scatterplot(x=data[numeric_columns[i]], y=data[numeric_columns[j]], hue=data['Hogwarts House'])
            plt.title(f'Scatter Plot: {numeric_columns[i]} vs {numeric_columns[j]}')
            plt.xlabel(numeric_columns[i])
            plt.ylabel(numeric_columns[j])
            plt.savefig(f'results_scatter/{numeric_columns[i]}_vs_{numeric_columns[j]}.png')
            plt.close()

if __name__ == "__main__":
    main()
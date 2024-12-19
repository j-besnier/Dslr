# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2024-12-19 11:56:52

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
import os
import textwrap


def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        print("Usage: histogram.py dataset.csv")
        sys.exit()
    
    data = pd.read_csv(sys.argv[1])
    data.replace('', np.nan, inplace=True)
    data.drop('Index', axis=1, inplace=True)
    numeric_columns = data.select_dtypes(include=np.number).columns
    
    houses = data['Hogwarts House'].unique()
    house_colors = {
        'Hufflepuff': '#FFD700',
        'Ravenclaw': '#4682B4',
        'Slytherin': '#2E8B57', 
        'Gryffindor': '#B22222'
    }
    
    if not os.path.exists('results_histograms'):
        os.makedirs('results_histograms')
    
    background_image = plt.imread('background.png')
    prop = fm.FontProperties(fname='ParryHotter.ttf')
    
    for column in numeric_columns:
        with PdfPages(f'results_histograms/{column}.pdf') as pdf:
            fig, ax = plt.subplots()
            ax.imshow(background_image, aspect='auto', extent=[0, 1, 0, 1], alpha=0.5, zorder=-1)
            wrapped_text = "\n".join(textwrap.wrap(column, width=40))
            ax.text(0.5, 0.5, wrapped_text, fontsize=25, ha='center', va='center', color='black', fontweight='bold', fontproperties=prop)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
            
            for house in houses:
                house_data = data[data['Hogwarts House'] == house]
                fig, ax = plt.subplots()
                house_data[column].dropna().hist(bins=30, color=house_colors.get(house, 'black'), ax=ax)
                ax.set_title(f'{house} - {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                pdf.savefig(fig)
                plt.close(fig)

if __name__ == "__main__":
    main()
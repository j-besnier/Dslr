# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2024-12-17 21:26:26

import pandas as pd
import sys
import numpy as np
import math
from tabulate import tabulate

def calculate_count(data, numeric_columns):
    result = ['Count']
    for column_name in numeric_columns:
        result.append(sum(data[column_name].notna()))
    return (result)
    
def calculate_mean(data, numeric_columns):
    result = ['Mean']
    for column_name in numeric_columns:
        Mean = data[column_name].notnull().sum() / sum(data[column_name].notna())
        result.append(str(Mean))
    return (result)

def calculate_std(data, numeric_columns):
    result = ['Std']
    for column_name in numeric_columns:
        column = data[column_name].dropna()
        Mean = sum(column) / len(column)
        Var = sum((column - Mean) ** 2) / len(column)
        std = Var ** 0.5
        result.append(str(std))
    return result
    
def calculate_min(data, numeric_columns):
    result = ['Min']
    for column_name in numeric_columns:
        column = data[column_name].dropna()
        min = column[0]
        for value in column:
            if (value < min):
                min = value
        result.append(str(min))
    return result
    
def calculate_max(data, numeric_columns):
    result = ['Max']
    for column_name in numeric_columns:
        column = data[column_name].dropna()
        max = column[0]
        for value in column:
            if (value > max):
                max = value
        result.append(str(max))
    return result
        
def calculate_quartile(data, numeric_columns, rank):
    result = [['25%'], ['50%'], ['75%']][rank - 1]
    for column_name in numeric_columns:
        column = data[column_name].dropna()
        ordered_column = column.sort_values(ignore_index = True)
        n = len(ordered_column)
        if rank == 1:
            result.append(str(ordered_column[math.ceil(n / 4) - 1]))
        elif rank == 2:
            result.append(str(ordered_column[math.ceil(n / 2) - 1]))
        elif rank == 3:
            result.append(str(ordered_column[math.ceil(3 * n / 4) - 1]))
    return result



def main():
    if (len(sys.argv) != 2):
        print("Wrong number of arguments")
        print("Usage: describe.py dataset.csv")
        sys.exit()
    data = pd.read_csv(sys.argv[1])
    data.replace('', np.nan, inplace=True)
    data.drop('Index', axis=1, inplace=True)
    numeric_columns = data.select_dtypes(include=np.number).columns
    print(tabulate([calculate_count(data, numeric_columns),
                    calculate_mean(data, numeric_columns),
                    calculate_std(data, numeric_columns),
                    calculate_min(data, numeric_columns),
                    calculate_quartile(data, numeric_columns, 1),
                    calculate_quartile(data, numeric_columns, 2),
                    calculate_quartile(data, numeric_columns, 3),
                    calculate_max(data, numeric_columns),
                    ], headers=([""] + [i for i in numeric_columns])))
    
    

if __name__ == "__main__":	
    main()

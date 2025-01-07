# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2025-01-07 21:10:16

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import sys

# Define the houses

# Dictionary to map house names to indices

def main():

    if len(sys.argv) != 3:
        print("Wrong number of arguments")
        print("Usage: evaluate.py dataset_truth.csv houses.csv")
        sys.exit()
    
    truth = pd.read_csv(sys.argv[1], header=0)['Hogwarts House']

    predictions = pd.read_csv(sys.argv[2] , header=0)['Hogwarts House']

    accuracy = accuracy_score(truth, predictions) * 100
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
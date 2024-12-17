# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2024-12-17 21:42:24

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    if (len(sys.argv) != 2):
        print("Wrong number of arguments")
        print("Usage: describe.py dataset.csv")
        sys.exit()
    data = pd.read_csv(sys.argv[1])
    data.replace('', np.nan, inplace=True)
    data.drop('Index', axis=1, inplace=True)


if __name__ == "__main__":	
    main()

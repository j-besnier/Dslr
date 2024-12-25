# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2024-12-25 22:43:56
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X @ weights)
    epsilon = 1e-5
    cost = (1/m) * ((-y.T @ np.log(h + epsilon)) - ((1 - y).T @ np.log(1 - h + epsilon)))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        weights = weights - (learning_rate/m) * (X.T @ (sigmoid(X @ weights) - y))
        cost_history[i] = compute_cost(X, y, weights)
    
    return weights, cost_history

def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        print("Usage: logreg_train.py dataset.csv")
        sys.exit()
    
    data = pd.read_csv(sys.argv[1])
    data.replace('', np.nan, inplace=True)
    data.drop('Index', axis=1, inplace=True)
    data.dropna(inplace=True)
    
    # Select numeric columns for features
    numeric_columns = data.select_dtypes(include=np.number).columns
    X = data[numeric_columns]
    
    # Target variable
    y = data['Hogwarts House']
    y = pd.get_dummies(y).values  # One-hot encoding of target variable
    
    

if __name__ == "__main__":
    main()
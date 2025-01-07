# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2025-01-07 21:19:36

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
from tqdm import tqdm


def normalize(df):
    return df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

def sigmoid(X):
        return np.array(1 / (1 + np.exp(-X)))

def create_one_hot(house, houses):
    return {h: int(h == house) for h in houses}

def validate_learning_rate(value):
    value = float(value)
    if value <= 0 or value >= 1:
        raise argparse.ArgumentTypeError(f"Learning rate must be between 0 and 1, got {value}")
    return value

def plot_costs(costs_dict):
    house_colors = ['#4682B4', '#2E8B57', '#B22222', '#FFD700']
    
    plt.figure(figsize=(10, 6))
    for (house, costs), color in zip(costs_dict.items(), house_colors):
        plt.plot(costs, label=house, color=color)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost vs. Epochs for All Houses')
    plt.legend()
    os.makedirs('results/results_regression', exist_ok=True)
    plt.savefig('results/results_regression/costs.png')
    plt.show()
    
class LogisticRegression:

    def __init__(self, theta=None, epochs=50000, learning_rate=0.0005, compute_costs=False):
        self.theta = np.array(theta)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.costs = []
        self.compute_costs = compute_costs

    def predict(self, x: np.array):
        X = np.insert(x, 0, 1, axis=1)
        return sigmoid(X.dot(self.theta))

    def cost(self, x, y):
        eps = 1e-15
        y_hat = self.predict(x)
        y = np.squeeze(y)
        cost = - sum((y * np.log(y_hat + eps)) + ((1 - y) * np.log(1 - y_hat + eps))) / len(y)
        return cost
        
    def fit(self, x, y, house):
        m = len(x)
        X = np.insert(x, 0, 1, axis=1)
        y = np.squeeze(y)
        for _ in tqdm(range(self.epochs), desc=f"Training for {house}"):
            gradient = 1 / m * (np.dot(X.T, (np.dot(X, self.theta) - y)))
            self.theta = self.theta - (self.learning_rate * gradient)
            if self.compute_costs:
                self.costs.append(self.cost(x, y))
        return self.theta
    

def main():
    
    parser = argparse.ArgumentParser(description='Train a logistic regression model.')
    parser.add_argument('input_file', type=str, help='Path to the input dataset CSV file')
    parser.add_argument('--c', action='store_true', help='Flag to calculate and store cost during training')
    parser.add_argument('--e', type=int, default=10000, help='Number of epochs for training')
    parser.add_argument('--lr', type=validate_learning_rate, default=0.05, help='Learning rate for training (between 0 and 1)')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: The file '{args.input_file}' does not exist.")
        sys.exit(1)
        

    if args.e > 100000 and args.c:
        response = input(f"Warning: You have set the cost calculation on and the number of epochs to {args.e}, it might take a LONG time to train. Do you want to continue? [y/N]: ")
        if response.lower() != 'y':
            print("Training aborted.")
            return
    
    df = pd.read_csv(args.input_file)
    df.fillna(method='ffill',inplace=True, axis=0)
    
    df = df.drop(['Birthday','First Name','Last Name', ], axis=1)
    df['Best Hand'] = df['Best Hand'].astype(str).map({'Left':'0','Right':'1'})
    df['Best Hand'] = pd.to_numeric(df['Best Hand'])
    
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    one_hot_encodings = {house: create_one_hot(house, houses) for house in houses}
    x = np.array(normalize(df.iloc[:, 2:]))
    thetas = []
    costs_dict = {}
    
    for house in houses:
        
        # Apply a mask on the labels for each house, for example for Ravenclaw, the mask will be [1, 0, 0, 0]
        y = np.array(df['Hogwarts House'].map(lambda h: one_hot_encodings[house][h]))
        
        model = LogisticRegression([1] * 15, epochs=args.e, learning_rate=args.lr, compute_costs=args.c)
        theta = model.fit(x, y, house)
        thetas.append(theta)
        
        if model.compute_costs:
            costs_dict[house] = model.costs
        
    if costs_dict:
        plot_costs(costs_dict)
    
    weights_df = pd.DataFrame(np.array(thetas).T, columns=['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'])
    os.makedirs('results/results_regression', exist_ok=True)
    weights_df.to_csv('results/results_regression/weights.csv', index=False)
    
if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
# @Author: Jean Besnier
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Jean Besnier
# @Last Modified time: 2025-01-07 21:10:55
import pandas as pd
import numpy as np
from logreg_train import LogisticRegression, normalize
import os
import sys



def main():
    if len(sys.argv) != 3:
        print("Wrong number of arguments")
        print("Usage: logreg_predict.py dataset_test.csv weights.csv")
        sys.exit()
    
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    index_to_house = {index: house for index, house in enumerate(houses)}

    weights_df = pd.read_csv(sys.argv[2])

    weights_dict = {col: weights_df[col].values for col in weights_df.columns}
        
    df_test = pd.read_csv(sys.argv[1])
    df_test.fillna(0,inplace=True)
    df_test = df_test.drop(['Birthday','First Name','Last Name'], axis=1)
    df_test['Best Hand'] = df_test['Best Hand'].astype(str).map({'Left':'0','Right':'1'})
    df_test['Best Hand'] = pd.to_numeric(df_test['Best Hand'])
    
    result = []
    x_test = np.array(normalize(df_test.iloc[:,2:]))
    for house in houses:
        model = LogisticRegression()
        model.theta = weights_dict[house]
        y_pred = model.predict(x_test)
        result.append(y_pred)
    result = np.c_[result].T
    
    prediction = np.argmax(result, axis=1)
    
    predicted_houses = [index_to_house[n] for n in prediction]

    predictions_df = pd.DataFrame({
        'Index': range(len(prediction)),
        'Hogwarts House': predicted_houses
    })

    os.makedirs('results/results_regression', exist_ok=True)
    predictions_df.to_csv('results/results_regression/houses.csv', index=False)


if __name__ == "__main__":
    main()
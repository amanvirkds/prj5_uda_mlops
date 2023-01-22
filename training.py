"""
Module: training.py
Author: Amandeep Singh
Date Written: 22-Jan-2023
Date Modified: 22-Jan-2023

Description:
    Module is for training the logistic regression model
    - module imports the data from specified source
    - process it
    - fit the logistic regression
    - saves the trained model to specified location
"""

import pickle
import os
import json
from flask import Flask, session, jsonify, request
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])

# Function for training the model


def train_model():
    """
    load the data, train logistic regression and save the model
    """

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to your data
    model_dataset = pd.read_csv(
        os.path.join(dataset_csv_path, "finaldata.csv"),
        dtype={
            "corporation": str,
            "lastmonth_activity": int,
            "lastyear_activity": int,
            "number_of_employees": int,
            "exited": int
        },
        usecols=[
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
            "exited"
        ]
    )
    X_train = model_dataset.loc[:, 
        ["lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"]
    ]
    y_train = model_dataset["exited"]
    model.fit(X_train, y_train)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    pickle.dump(
        model,
        open(
            os.path.join(model_path, 'trainedmodel.pkl'),
            'wb'
        )
    )


if __name__ == "__main__":
    train_model()

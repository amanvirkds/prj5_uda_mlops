"""
Module: scoring.py
Author: Amandeep Singh
Date Written: 22-Jan-2023
Date Modified: 22-Jan-2023

Description:
    Module is for testing the model on test data and record the
    metric
    - module imports the model and test data
    - predict the values using model
    - calculates the f1 metric and save it to file "latestscore.txt"
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

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

# Function for model scoring


def score_model():
    """
    this function should take a trained model, load test data, and calculate an
    F1 score for the model relative to the test data.
    write the result to the latestscore.txt file
    """

    # load model
    model = pickle.load(
        open(
            os.path.join(model_path, 'trainedmodel.pkl'),
            'rb'
        )
    )

    # read test data
    test_data = pd.read_csv(
        os.path.join(test_data_path, "testdata.csv"),
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

    # predict and calculate the f1 metric
    preds = model.predict(
        test_data.loc[:,
                      ["lastmonth_activity",
                       "lastyear_activity",
                       "number_of_employees"]
                      ]
    )
    f1_score = metrics.f1_score(
        test_data["exited"],
        preds
    )

    # write the latest f1 score to the file
    with open(
        os.path.join(model_path, "latestscore.txt"), "w"
    ) as scoring_file:
        scoring_file.write(str(f1_score))

    return f1_score


if __name__ == "__main__":
    score_model()

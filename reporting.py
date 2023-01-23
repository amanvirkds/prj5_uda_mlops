
"""
Module: reporting.py
Author: Amandeep Singh
Date Written: 23-Jan-2023
Date Modified: 23-Jan-2023

Description:
    - Module to calculate the predictions, generate confusion matrix and
     save it to directory
"""

import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from diagnostics import model_predictions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for reporting
def score_model():
    """
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """

    # read test data
    data = pd.read_csv(
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
    # load model
    model = pickle.load(
        open(
            os.path.join(model_path, 'trainedmodel.pkl'),
            'rb'
        )
    )

    preds = model_predictions()
    cm = metrics.confusion_matrix(data.exited, preds)

    # Setting default size of the plot
    # Setting default fontsize used in the plot
    plt.rcParams['figure.figsize'] = (10.0, 9.0)
    plt.rcParams['font.size'] = 20

    # Implementing visualization of Confusion Matrix
    display_c_m = metrics.ConfusionMatrixDisplay(cm)

    # Plotting Confusion Matrix
    # Setting colour map to be used
    display_c_m.plot(cmap='OrRd', xticks_rotation=25)
    # Setting fontsize for xticks and yticks
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Giving name to the plot
    plt.title('Confusion Matrix', fontsize=24)

    # Saving plot
    plt.savefig(
        os.path.join(model_path, 'confusion_matrix.png'),
        transparent=True,
        dpi=500
    )


if __name__ == '__main__':
    score_model()

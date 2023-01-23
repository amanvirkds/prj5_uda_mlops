"""
Module: diagnostics.py
Author: Amandeep Singh
Date Written: 23-Jan-2023
Date Modified: 23-Jan-2023

Description:
    - Module performs following diagnostics
    -- model_predictions:
        returns the predicitons on given dataset using saved model

    -- dataframe_summary:
        returns the summary statistics of each numeric column

    -- dataframe_missing_values:
        returns pct of missing values of each columns as list

    -- execution_time:
        returns a list of execution time of training.py and ingestion.py

    -- outdated_packages_list:
        returns the list of outdated packages
"""

import os
import json
import subprocess
import pickle
import pandas as pd
import numpy as np
import timeit


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

# Function to get model predictions


def model_predictions():
    """
    read the deployed model and a test dataset, calculate predictions
    """

    # load model
    model = pickle.load(
        open(
            os.path.join(model_path, 'trainedmodel.pkl'),
            'rb'
        )
    )
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
    # predict the values
    preds = model.predict(
        data.loc[:,
                      ["lastmonth_activity",
                       "lastyear_activity",
                       "number_of_employees"]
                      ]
    )

    return preds

# Function to get summary statistics


def dataframe_summary():
    """
    calculate the summary statistics of a dataset
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
        ]
    )
    summary_statistics_list = []
    for col in data.columns:
        stats = []
        stats.append(data[col].mean())
        stats.append(data[col].median())
        stats.append(data[col].std())
        summary_statistics_list.append(stats)

    # return value should be a list containing all summary statistics
    return summary_statistics_list


def dataframe_missing_values():
    """
    calculate the missing values in a dataset
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
        }
    )
    missing_pct_list = [
        missing / data.shape[0] for missing in data.isna().sum().tolist()
    ]

    # return value should be a list containing all summary statistics
    return missing_pct_list

# Function to get timings


def execution_time():
    """
    calculate timing of training.py and ingestion.py
    """

    execution_times_list = []

    starttime = timeit.default_timer()
    os.system("python training.py")
    timing = timeit.default_timer() - starttime
    execution_times_list.append(timing)

    starttime = timeit.default_timer()
    os.system("python ingestion.py")
    timing = timeit.default_timer() - starttime
    execution_times_list.append(timing)

    return execution_times_list

# Function to check dependencies


def outdated_packages_list():
    """
    to get the list of outdated packages
    """

    outdated = subprocess.check_output(
        ["pip", "list", "--outdated"]
    )

    return outdated


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    dataframe_missing_values()
    execution_time()
    outdated_packages_list()

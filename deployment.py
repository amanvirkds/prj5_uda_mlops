"""
Module: deployment.py
Author: Amandeep Singh
Date Written: 22-Jan-2023
Date Modified: 22-Jan-2023

Description:
    Module is for deploying the model to production enviornment
    - moves the model, score and data to production enviornment
"""

import pickle
import os
import json
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# function for deployment
def store_model_into_pickle(model):
    """
    copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt
    file into the deployment directory
    """

    # save the model to deployment path
    pickle.dump(
        model,
        open(
            os.path.join(prod_deployment_path, 'trainedmodel.pkl'),
            'wb'
        )
    )

    # save the scores
    with open(os.path.join(model_path, "latestscore.txt"), "r") as f:
        contents = f.read()
    with open(
        os.path.join(prod_deployment_path, "latestscore.txt"), "w"
    ) as scoring_file:
        scoring_file.write(contents)

    # save the data file
    final_df = pd.read_csv(
        os.path.join(dataset_csv_path, "finaldata.csv"),
        dtype={
            "corporation": str,
            "lastmonth_activity": int,
            "lastyear_activity": int,
            "number_of_employees": int,
            "exited": int
        }
    )
    final_df.to_csv(
        os.path.join(prod_deployment_path, "finaldata.csv"), index=False
    )


if __name__ == "__main__":
    # load model
    model = pickle.load(
        open(
            os.path.join(model_path, 'trainedmodel.pkl'),
            'rb'
        )
    )
    store_model_into_pickle(model)

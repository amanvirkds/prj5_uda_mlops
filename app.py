"""
Module: reporting.py
Author: Amandeep Singh
Date Written: 23-Jan-2023
Date Modified: 23-Jan-2023

Description:
    - Flask based api with following endpoints
    -- 
        - /prediction
            - to get the model prediciton on given data
        - /scoring
            - to get the latest model scoring value
        - /summarystats
            - to get the summary statistics of dataset
        - /diagnostics
            - to get the missing values, execution time, outdated packages
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis 
#import predict_exited_from_saved_model
import json
import os
from diagnostics import (model_predictions, dataframe_summary, 
                        dataframe_missing_values, execution_time, outdated_packages_list)
from scoring import score_model



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        

    #call the prediction function you created in Step 3
    data = pd.read_csv(
        os.path.join(dataset_csv_path, "finaldata.csv"),
        dtype={
            "corporation": str,
            "lastmonth_activity": int,
            "lastyear_activity": int,
            "number_of_employees": int,
            "exited": int
        }
    )
    preds = model_predictions(data)

    return str(preds)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():  
    """
    Scoring endpoint that runs the script scoring.py and
    gets the score of the deployed model

    Returns:
        str: model f1 score
    """      
    score = score_model()

    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    """
    Summary statistics endpoint that calls dataframe summary
    function from diagnostics.py

    Returns:
        json: summary statistics
    """

    summary_stats = dataframe_summary()

    return str(summary_stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnosis():        
    """
    Diagnostics endpoint thats calls missing_percentage, execution_time,
    and outdated_package_list from diagnostics.py

    Returns:
        dict: missing percentage, execution time and outdated packages
    """

    #check timing and percent NA values
    data = pd.read_csv(
        os.path.join(dataset_csv_path, "finaldata.csv"),
        dtype={
            "corporation": str,
            "lastmonth_activity": int,
            "lastyear_activity": int,
            "number_of_employees": int,
            "exited": int
        }
    )
    missing_values = str(dataframe_missing_values())
    execution = str(execution_time())
    outdated = str(outdated_packages_list())

    return {
        'missing_percentage': missing_values,
        'execution_time': execution,
        'outdated_packages': outdated
    }


if __name__ == "__main__":    
    app.run(
        host='0.0.0.0', 
        port=8000, 
        debug=True, 
        threaded=True
    )


"""
Module: fullprocess.py
Author: Amandeep Singh
Date Written: 24-Jan-2023
Date Modified: 24-Jan-2023

Description:
    - Module peforms following actions
        - check if there are any new files in source dir
            - if there are new files perform data ingestion
        - if there are new files, then retrain the model
            - if new model score is different then the old score
        - then redploy and perform the diagnosis
"""

import os
import json
import pickle
import logging
import pandas as pd
from sklearn import metrics

import ingestion
import training
import scoring
import deployment
import apicalls
import reporting

RETRAIN_FLAG = False

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = config['output_folder_path']
test_data_path = os.path.join(config['test_data_path'])

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(asctime)s %(process)d %(message)s',
    filename=os.path.join(os.getcwd(), 'logs/process.log'),
    filemode='w')

# Check and read new data
# first, read ingestedfiles.txt

logging.info("Read ingested files record!")
ingestion_records_df = pd.read_csv(
    os.path.join(
        prod_deployment_path,
        'ingestedfiles.txt'
    ),
    names=[
        'source_dir',
        'filename',
        'records',
        'date',
        'undef'
    ]
)

# second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt

new_files = []

for file in os.listdir(os.path.join(
        os.getcwd(), input_folder_path)):
    if ((file not in ingestion_records_df.filename.tolist())
            & (file[-4:] == ".csv")):
        new_files.append(file)

if len(new_files) > 0:
    logging.info(f"{len(new_files)} new files!")
    RETRAIN_FLAG = True
else:
    logging.info("No new files, stoping the process!")


# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here

if RETRAIN_FLAG:
    logging.info("Ingesting new data!")
    ingestion.merge_multiple_dataframe()

# Checking for model drift
# check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data

# read the f1 score for last model training
with open(
    os.path.join(prod_deployment_path, "latestscore.txt"), "r"
) as scoring_file:
    last_f1_score = scoring_file.read()
if RETRAIN_FLAG:
    logging.info("Validate the model scores!")

    # read test data
    new_data = pd.read_csv(
        os.path.join(output_folder_path, "finaldata.csv"),
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
    # predict and calculate the f1 metric
    preds = model.predict(
        new_data.loc[:,
                     ["lastmonth_activity",
                      "lastyear_activity",
                      "number_of_employees"]
                     ]
    )
    new_f1_score = metrics.f1_score(
        new_data["exited"],
        preds
    )

    # check model drift
    if last_f1_score == new_f1_score:
        logging.info("No change in model scores!")
        RETRAIN_FLAG = False
    else:
        logging.info("Change in model scores!")

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the
# process here

if RETRAIN_FLAG:
    logging.info("Retrain and score the model!")
    training.train_model()
    scoring.score_model()

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script


if RETRAIN_FLAG:
    logging.info("Redployment of the model!")
    model = pickle.load(
        open(
            os.path.join(model_path, 'trainedmodel.pkl'),
            'rb'
        )
    )
    deployment.store_model_into_pickle(model)

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model

if RETRAIN_FLAG:
    logging.info("Perform model diagnosis and reporting!")
    apicalls.api_calls()
    reporting.score_model()

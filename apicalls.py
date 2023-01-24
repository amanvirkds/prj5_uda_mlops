"""
Module: reporting.py
Author: Amandeep Singh
Date Written: 23-Jan-2023
Date Modified: 23-Jan-2023

Description:
    - test the api calls
"""

import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


def api_calls():
    #Call each API endpoint and store the responses
    response1 = requests.post(
        f'{URL}/prediction',
        json={
            'filepath': os.path.join(test_data_path, 'testdata.csv')}).text
    response2 = requests.get(f'{URL}/scoring').text
    response3 = requests.get(f'{URL}/summarystats').text
    response4 = requests.get(f'{URL}/diagnostics').text

    #combine all API responses
    responses = (
        "Model Predictions: \n"
        + response1 + "\n\n" 
        "Scoring: \n"
        + response2 + "\n\n" 
        "Summary Stats: \n"
        + response3 + "\n\n" 
        "Diagnosis: \n"
        + response4)

    #write the responses to your workspace
    with open(os.path.join(model_path, 'apireturns.txt'), 'w') as file:
        file.write(responses)

if __name__ == '__main__':
    api_calls()



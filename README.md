<h2>Introduction<h3>

<div><p>
Logistic Regression model to estimate the attrition risk of each of the company's 10000 clients

- Model is using MLOps practices to auto-retrain if there is change in data or model performance

</p></div>

<h2>Steps in Modeling process</h2>

Following steps are performed by the process:

<h3>Ingestion</h3>

- look for the csv files in input directory
- read the files and create a single dataset
- drop the duplicates
- write the final dataset to output directory
- keep the processing record in ingestedfiles.txt

<h3>Training</h3>

- imports the data from specified source
- process it
- fit the logistic regression
- saves the trained model to specified location

<h3>Scoring</h3>

- Imports the model and test data
- predict the values using model
- calculates the f1 metric and save it to file "latestscore.txt"

<h3>Diagnosis</h3>

- predicitons on given dataset using saved model
- summary statistics of each numeric column
- percentage of missing values of each columns as list
- calculates execution time of training.py and ingestion.py
- checks the list of outdated packages

<h3>Reporting<h3>

- calculate the predictions, generate confusion matrix and save it to directory

<h3>API calls<h3>

- API Calls

<h3>Full Process<h3>

- Looks if there is change in data
- Retrain and redeploy the model

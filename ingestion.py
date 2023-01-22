"""
Module: ingestion.py
Author: Amandeep Singh
Date Written: 22-Jan-2023
Date Modified: 22-Jan-2023

Description:
    Module is to read the raw data from specified directory and performs the processing
    - look for the csv files in input directory
    - read the files and create a single dataset
    - drop the duplicates
    - write the final dataset to output directory
    - keep the processing record in ingestedfiles.txt

"""

from datetime import datetime
import os
import json
import pandas as pd


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
record_datasource_path = config['record_datasoruce_path']
RECORD_DATASORUCE_FILE = "ingestedfiles.txt"


# Function for data ingestion
def merge_multiple_dataframe():
    """
    check for datasets, compile them together, and write to an output file
    """

    # look for the csv files in sepecified path location
    # read each file and then combine them to single dataset
    dfs = []
    records_list = []
    for file in os.listdir(input_folder_path):
        if file[-4:] == ".csv":
            data = pd.read_csv(
                os.path.join(input_folder_path, file)
            )

            # add to dataframes list
            dfs.append(data)

            # record the details of ingested file
            date_time_obj = datetime.now()
            thetimenow = (
                str(date_time_obj.year)
                + str(date_time_obj.month)
                + str(date_time_obj.day)
            )
            records = [
                input_folder_path,
                file,
                len(data.index),
                thetimenow
            ]
            records_list.append(records)

    with open(
        os.path.join(record_datasource_path, RECORD_DATASORUCE_FILE), "w"
    ) as record_file:
        for record in records_list:
            for element in record:
                record_file.write(str(element) + ",")
            record_file.write('\n')

    # combine all the imported dataframes
    final_df = pd.concat(dfs)

    # deduplication of the data
    final_df.drop_duplicates(keep="first", inplace=True)

    # write the processed data for further processing
    final_df.to_csv(
        os.path.join(output_folder_path, "finaldata.csv"), index=False
    )


if __name__ == '__main__':
    merge_multiple_dataframe()

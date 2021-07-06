# importing necessary libraries
import pickle
from pickle import dumps
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from os import path
from os import listdir
from os.path import isfile, join
import os
import mlrun
from pymongo import MongoClient
import urllib
import sys
import pandas as pd
import pymongo
import json
import os

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

from .mlPipeline import SuicideModel


# getting current working directory
cwd = os.getcwd()
serving = None



# function for running whole MLRUN pipeline
def mlRunPipeline():
    global serving
    suicide_func = mlrun.code_to_function(name='suicide', kind='job', filename = cwd +'/ml_model/mlPipeline.py') # converting code to fucntion from mlrun pipeline

    # fetching data
    fetch_data_run = suicide_func.run(handler='fetch_data',
                                inputs={'data_path': 'suicidal_data1.csv'},
                                local=True)

    print(fetch_data_run.outputs)
    
    # transforming data
    transform_dataset_run = suicide_func.run(name='transform_dataset',
                                        handler='transform_dataset',
                                        inputs={'data': fetch_data_run.outputs['suicide_dataset']},
                                        local=True)

    print(transform_dataset_run.outputs)
    
    # trainging model
    train_model_run = suicide_func.run(name='train_model',
                                    handler='train_model',
                                    inputs={'input_ds': transform_dataset_run.outputs['suicide_dataset_transformed']},
                                    local=True)

    print(train_model_run.outputs)

    # serving the running model
    serving = mlrun.code_to_function('serving', filename=cwd +'/ml_model/mlPipeline.py', kind='serving')

    serving.spec.default_class = 'SuicideModel'
    serving.add_model('suicide-serving', train_model_run.outputs['Suicide_Model'])
    


    
# updating the database by uploading info from local db to mongodb atlas
def updateDatabase(filePath):
    file = filePath
    m_client = pymongo.MongoClient("<Enter mongodb atlas client link here>")
    db = m_client.test
    #m_client = pymongo.MongoClient("mongodb://...")
    m_db = m_client["suicide"]
    db_cm = m_db["suicide"]


    data = pd.read_csv(file)
    data_json = json.loads(data.to_json(orient='records'))
    print(db_cm.find() )
    i = 0
    for data in data_json:
        i = i+1
        print(i)
        business = {"value":i}
        db_cm.insert_one(data)


# prediction function to predict over the input using the serving
def pred(text):
    print("Text Received =>", text)
    # input text from the UI
    input_text = text
    text = {"inputs":[text]}
    
    # loading the serving
    server = serving.to_mock_server()
    pred = server.test("/v2/models/suicide-serving/infer", body=text)
    status = "Suicidal" if pred['outputs'][0] == 1 else "Neutral "

    data_dir = cwd + "/ml_model/data"
    
    # model retraining by specifying a particular limit
    if ('retrain.csv' in [f for f in listdir(data_dir) if isfile(join(data_dir, f))]):
        print("retrain.csv available for training")
        file = data_dir + "/retrain.csv"
        df = pd.read_csv(file)
        if len(df) > 4:
            updateDatabase(file)
            df = pd.DataFrame(columns=["label", "tweet"])
            df.to_csv(cwd + "/ml_model/data/retrain.csv", index=False)
            mlRunPipeline()
        else:
            print(df)
            data = {
                "label": pred['outputs'][0],
                "tweet": input_text
            }
            
            df = df.append(data, ignore_index=True)
            print("change", data, df)
            df.to_csv(cwd + "/ml_model/data/retrain.csv", index=False)

    else:
        data = {
                "label": [pred['outputs'][0]],
                "tweet": [input_text]
            }
        df = pd.DataFrame(data)
        df.to_csv(cwd + "/ml_model/data/retrain.csv", index=False)

    return status


# function for bulktraining the data from uploaded file
def bulktraining(file):
    
    updateDatabase(file)
    mlRunPipeline()

    return True

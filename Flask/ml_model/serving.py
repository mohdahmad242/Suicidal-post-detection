
import pickle
from pickle import dumps
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from os import path
import os
import mlrun

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

from .mlPipeline import SuicideModel

suicide_func = mlrun.code_to_function(name='suicide', kind='job', filename = '/media/ahmadkhan242/A/MLRUn/Flask/ml_model/mlPipeline.py')


fetch_data_run = suicide_func.run(handler='fetch_data',
                               inputs={'data_path': 'suicidal_data1.csv'},
                               local=True)

print(fetch_data_run.outputs)

print("###############################################")

transform_dataset_run = suicide_func.run(name='transform_dataset',
                                      handler='transform_dataset',
                                      inputs={'data': fetch_data_run.outputs['suicide_dataset']},
                                      local=True)

print(transform_dataset_run.outputs)

train_model_run = suicide_func.run(name='train_model',
                                handler='train_model',
                                inputs={'input_ds': transform_dataset_run.outputs['suicide_dataset_transformed']},
                                local=True)

print(train_model_run.outputs)


serving = mlrun.code_to_function('seving', filename='/media/ahmadkhan242/A/MLRUn/Flask/ml_model/mlPipeline.py', kind='serving')

serving.spec.default_class = 'SuicideModel'
serving.add_model('suicide-serving', train_model_run.outputs['Suicide_Model'])
# serving_address = serving.deploy()




my_data = '''{"inputs":["It's such a hot day, I'd like to have ice cream and visit the park"]}'''

server = serving.to_mock_server()
print(server.test("/v2/models/suicide-serving/infer", body=my_data))



def pred(text):
    print("Text Received =>", text)

    text = {"inputs":[text]}
    
    server = serving.to_mock_server()
    pred = server.test("/v2/models/suicide-serving/infer", body=text)

    status = "Suicidal" if pred['outputs'][0] == 1 else "Neutral "
    return status

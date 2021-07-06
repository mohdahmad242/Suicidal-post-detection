# importing necessary libraries for the code
from mlrun import mlconf
from os import path
import mlrun
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
from pymongo import MongoClient
import urllib
import sys
import pandas as pd
import pymongo
import json
import os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from cloudpickle import load
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# diwnloading stopwords corpus from nltk
nltk.download('stopwords')
stop = stopwords.words('english')


# get current working directory
cwd = os.getcwd()


# configuring dbpath and artifactpath
mlconf.dbpath = cwd + '/MLRun_data'
mlconf.artifact_path = cwd + '/MLRun_data/Data'


# settiing project name base
project_name_base = 'suicide-pred'

# setting mlrun environment
project_name, artifact_path = mlrun.set_environment(project=project_name_base, user_project=True)


# printing projectname and artifactpath
print(f'Project name: {project_name}')
print(f'Artifact path: {artifact_path}')



# THE PIPELINE CODE IS STARTED FROM BELOW COMMENT OF THE START CODE


# nuclio: start-code

def read_content():
    m_client = pymongo.MongoClient("<Paste your mongodb atlas connection link here>") # initializing mongoclient
    db = m_client.test # testing mongoclient

    m_db = m_client["suicide"] # collection name
    db_cm = m_db["suicide"] # database name
    df = pd.DataFrame.from_records(db_cm.find()) # reading data from mongodb atlas
    print('--------------------------------')
    print()
    print(df)
    return df # returning dataframe


df=read_content() # getting dataframe from read_content function
df=df[['label','tweet']]
df.to_csv('suicidal_data1.csv') # saving the csv

# fetching data and keeping logs for the data for MLRUN
def fetch_data(context : MLClientCtx, data_path: DataItem):
    
    context.logger.info('Reading data from {}'.format(data_path))

    suicide_dataset = df
    
    target_path = path.join(context.artifact_path, 'data')
    context.logger.info('Saving datasets to {} ...'.format(target_path))

    # Store the data sets in your artifacts database
    context.log_dataset('suicide_dataset', df=suicide_dataset, format='csv',
                        index=False, artifact_path=target_path)

# preprocessing the data for the MLRUN
def preprocess_tweet(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text+' '.join(emoticons).replace('-', '') 
    return text


# stemmer to stem the data for MLRUN
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]



# tokenizing the data
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\(|D|P)',text.lower())
    text = re.sub('[\W]+', ' ', text.lower())
    text += ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in tokenizer_porter(text) if w not in stop]
    return tokenized



vect = HashingVectorizer(decode_error='ignore', n_features=2**21, 
                         preprocessor=None,tokenizer=tokenizer)

# transforming the dataset for the MLRUN    
def transform_dataset(context : MLClientCtx, data: DataItem):

  context.logger.info('Begin datasets transform')

  df = data.as_df()
  df['tweet'] = df['tweet'].apply(lambda x: preprocess_tweet(x))

  target_path = path.join(context.artifact_path, 'data')
  context.log_dataset('suicide_dataset_transformed', df=df, artifact_path=target_path, format='csv')


# training the data in MLRUN pipeline
def train_model(context: MLClientCtx, input_ds: DataItem):
  context.logger.info('Begin training')
  from sklearn.linear_model import SGDClassifier
  from sklearn.linear_model import Perceptron

  clf1 = SGDClassifier(loss='log', random_state=1) # initializing SGD classifier
  clf2 = Perceptron(tol=1e-3, random_state=0) # initializing Perceptron model


  df = input_ds.as_df()
  X = df["tweet"].to_list()
  y = df['label']

  from sklearn.model_selection import train_test_split
  X_train,X_test,y_train,y_test = train_test_split(X,
                                                  y,
                                                  test_size=0.20,
                                                  random_state=0)  
  X_train = vect.transform(X_train)
  X_test = vect.transform(X_test)

  classes = np.array([0, 1])
  clf1.partial_fit(X_train, y_train,classes=classes) # partially fitting data for SGD
  clf2.partial_fit(X_train, y_train,classes=classes) # partially fitting data for perceptron

  acc1 = clf1.score(X_test, y_test) # getting accuracy score for SGD
  acc2 = clf2.score(X_test, y_test) # getting accuracy score for perceptron

  print('Accuracy SGD: %.3f' % acc1)
  print('Accuracy Perceptron: %.3f' % acc2)

  context.log_result("accuracyon SGD Model", acc1*100 ) # logging accuracy for SGD
  context.log_result("accuracy on Perceptron Model", acc2*100 ) # logging accuracy for perceptron
  
  # taking the model with best accuracy for SGD and perceptron 
  if acc1>=acc2:
    context.log_model('Suicide_Model',
                      body=dumps(clf1),
                      artifact_path=context.artifact_subpath("models"),
                      model_file="Suicide_Model.pkl")
    context.logger.info('Model Selected : SGD')  
    

  if acc1<acc2:
    context.log_model('Suicide_Model',
                      body=dumps(clf2),
                      artifact_path=context.artifact_subpath("models"),
                      model_file="Suicide_Model.pkl")
    context.logger.info('Model Selected: Perceptron')
    


  context.logger.info('End training') # log of training end



# Suicide model class for loading data and predicting
class SuicideModel(mlrun.serving.V2ModelServer):
    def load(self):
        model_file, extra_data = self.get_model('.pkl')
        self.model = load(open(model_file, 'rb'))

    def predict(self, body):
        try:
            feats = body['inputs'][0]
            # feats = feats.decode('ISO-8859-1')
            feats = preprocess_tweet(feats)
            l = []
            l.append(feats)
            feats = vect.transform(l)
            result = self.model.predict(feats)
            return result.tolist()
        except Exception as e:
            raise Exception("Failed to predict %s" % e)

# nuclio: end-code



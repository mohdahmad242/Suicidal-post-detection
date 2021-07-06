# Suicidal post detection
[Deployed on Heroku]()
*** 
## Structure
``` bash
─── Suicidal post detection.  
    ├── Notebook  
        ├── mlrun_pipeline.ipyub (experiment log included)
    ├── WebApp  
    ├── requirement.txt  
```
## Overview 
## Inspiration
Suicide might be considered as one of the most serious social health problems in the modern society. Every year, almost 800,000 people commit suicide. Suicide remains the second leading cause of death among a young generation with an overall suicide rate of 10.5 per 100,000 people. Specially in this pandemic suicide rate has increased. Many factors can lead to suicide, for example, personal issues, such as hopelessness, severe anxiety, or impulsivity; social factors, like social isolation and overexposure to deaths; or negative life events, and previous suicide attempts, etc. Millions of peoples around the world fall victims to suicide every year, making suicide prevention become a critical global public health mission.

## What it does
Early detection and treatment are regarded as the most effective ways to prevent suicidal ideation and potential suicide attempts—two critical risk factors resulting in successful suicides.   
In this project we propose an API which can be used to tag any textual data with a potential suicidal thought tag or neutral tag. 
We used supervised Machine learning to train our model which can detect Suicidal severity of the post.  
We also deployed our model on **Heroku** using **Flask** and **mlrun** alongwith **MongoDB Atlas** to demonstrate the **`applicability`** of the project.

## How we built it
We took tweet dataset published on this  [Github repository](https://github.com/AminuIsrael/Predicting-Suicide-Ideation). We trained our model on two different model SGD classifier and simple preceptor model. 
We used `mlrun` to automate our pipeline. We first fetch dataset from MongoDB Atlas database, pre-process it, and trained two model discussed earlier, and finally best model is used for serving.   
We also deployed it on `Heroku` to display the `applicability` of the API. Our pipeline is fully automated and robust to data accusation, when someone use our API, it store each instance in CSV file on the server and once it reached a limit, the pipeline automatically push the data to MongoDB database and retrain the models. Also, choose the best model out of two for severing, so basically we are doing Semi-Supervise learning to make our model better.
### Pipeline workflow
For the pileline automation and tracking of logs, we used an open sourced **`mlrun`** library which give us the flexibility to create Machine learning pipeline, manage the pipeline logs, and deploy it in production environment.   
The features we laveraged from this library are automates data fetching and prepration, model training and testing, deployment of real-time production pipelines, and end-to-end monitoring using heroku server logs.  
Image below depicts our `MLRun pipeline workflow`

<p align="center">
 <a href="https://imdbmovienew.herokuapp.com/"><img src="https://github.com/ahmadkhan242/Suicidal-post-detection/blob/main/images/WorkFlow.png" style="width: auto; max-width: 100%; height: auto" title="Web Application" /></a>
</p> 

We successfully implemented the following process with help of **`mlrun`**.  
### Fetching data.   
Data are stored in MongoDB atlas on cloud databse. When the mlrun pipeline is executed the fetch function will download the suicide dataset from MongoDB atlas database, convert it into DataFrame and serve it for data transformation.  
    <p align="center">
 <a href="https://imdbmovienew.herokuapp.com/"><img src="https://github.com/ahmadkhan242/Suicidal-post-detection/blob/main/images/fetchData.png" style="width: auto; max-width: 100%; height: auto" title="Web Application" /></a>
</p> 

### Transformation of data.  
The process of pre-processing is added to tranform function of the pipeline, we took advantage of the mlrun's `PlotArtifacts` API to plot some insight of the dataset. Finally the function pre-process our dataset and serve it for model training.   
### Model training and evaluation.
In this pipeline we training our data on different models
1. SCD classifier
2. Preceptor model
  <p align="center">
 <a href="https://imdbmovienew.herokuapp.com/"><img src="https://github.com/ahmadkhan242/Suicidal-post-detection/blob/main/images/modelFlow.png" style="width: auto; max-width: 100%; height: auto" title="Web Application" /></a>
</p> 

The dataset is trained on both model, then after evaluation the accuracy is compared in the pipeline. And the best model is saved and used for serving with the help of `mlrun`.  
The logs of the model training were tracked bu `mlrun` for both model. Here also we took advantage of the mlrun's `PlotArtifacts` API to plot the accuracies of the model. Below is the example of log.  

   <p align="center">
 <a href="https://imdbmovienew.herokuapp.com/"><img src="https://github.com/ahmadkhan242/Suicidal-post-detection/blob/main/images/training.png" style="width: auto; max-width: 100%; height: auto" title="Web Application" /></a>
</p> 

### Model serving.  
After pre-processing and model training , best model was selected by the pipeline and used for inference. With help of `mlrun` with serve our model for testing. Example is dmonstrated below.
<p align="center">
 <a href="https://imdbmovienew.herokuapp.com/"><img src="https://github.com/ahmadkhan242/Suicidal-post-detection/blob/main/images/serving.png" style="width: auto; max-width: 100%; height: auto" title="Web Application" /></a>
    
### Model retraining.  
This one of the main feature of our pipeline. We integrated **`retraining`** of our models using two different processe-
1. Inference time data: We collected the data produce during inference time, we stored the input data and predicted label on our server in `csv` file. Once the data reaches to a limit of 1000 samples. Our pipeline push the dataset to `MongoBD atlas` on cloud and begin the retraining process, discussed earlier.
2. Bulk training: We also integrated bulk training in our pipeline. When we collect dataset from other sources in bulk, we can use `/bulktraining` API. We can pass csv file, our pipeline will store the dataset on cloud and begin the retraining process, discussed earlier.
    
## Accomplishments that we're proud of
* Our project demonstrate a robust pipeline which is automated with help of `mlrun` library.
* Our project has multiple independent steps for data acquisition from **`MongoDB Atlas`** database, pre-processing , and training of the model.
* We also include multiple models in our project for training so as to get better model for serving.
* For data acquisition, we included a functionalists in our pipeline to retrain the model once a certain amount of data is collected.
* We incoporated **bulk training facility** in our pipeline.
* Our pipeline is deployed on `Heroku` in real time with help of `Flask` and `mlrun` library.
* We included our notebook which will help other to convert their project into `MLRUN` pipeline.
* We also include all code and documentation on Github on how to deploy the pipeline on Heroku.

## What's next for Suicide predictions 
This project is not limited to only Suicide post prediction, in this new age social media world, textual data is being generated every second. These data can be leveraged in many ways, like deep sentiment analysis, health care related problems can be solved, hate and toxicity of the post can be detected to stop bullying, etc. 

## Installation and Execution.
- Clone the repository using `git clone` and then change the directory to root of the project
``` 
    git clone https://github.com/ahmadkhan242/Suicidal-post-detection.git
    cd Suicidal-post-detection
```
- Create a virtual Environament, activate it and install requirement.txt file
```
> virtualenv venv

> source ./venv/Scripts/activate 

> pip install -r requirement.txt
```
- For Jupyter Notebook
```
> cd Notebook
> jupyter lab or jupyter notebook
```
- For WebApp
```
> cd WebApp
> flask run
```
> Website Live at `http://127.0.0.1:5000/`.  

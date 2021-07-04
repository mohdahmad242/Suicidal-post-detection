# Suicidal post detection
[Deployed on Heroku]()
> This a project on suicidal post detection
*** 
## Structure
``` bash
─── Suicidal post detection.  
    ├── Notebook  
        ├── mlrun_pipeline.ipyub (experiment log included)
        ├── flair_dataset.csv
    ├── WebApp  
    ├── requirement.txt  
```
## Overview 

## Installation and Execution.
- Clone the repository using `git clone` and then change the directory to root of the project
``` 
    git clone https://github.com/ahmadkhan242/Suicidal-post-detection.git
    cd Reddit-flair-detection
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

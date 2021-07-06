        
 <br>
 <h2 align="center">Deployment of Machine Learning Model on Heroku using Flask</h2>

 <p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/badge/python-3%2B-brightgreen?logo=Python">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/badge/git-2.29.2-brightgreen?logo=git">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Heroku" src="http://img.shields.io/static/v1?label=Pytorch&message=1.6.0&color=brightgreen&logo=Pytorch">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Heroku" src="http://img.shields.io/static/v1?label=Flask&message=1.1.1&color=brightgreen&logo=Flask">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Heroku" src="http://img.shields.io/static/v1?label=Postman&message=tested&color=brightgreen&logo=Postman">
    </a>
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Heroku" src="http://img.shields.io/static/v1?label=Heroku&message=Deployed&color=brightgreen&logo=Heroku">
    </a>
</p>
 <br>

## Introduction
Flask is a lightweight web framework written in Python. Flask is easy to use, and to get started for beginners. It is classified as a microframework because it does not require particular tools or libraries to work. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.  
In this section of the tutorial, you will learn how to set up a Flask project and to deploy a Machine Learning model you have developed in the [`previous chapter`](https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/README.md). By the end of this section you will be able to deploy any model using `Flask` on `Heroku`.

> All code files for this project are availabe here - https://github.com/ahmadkhan242/Suicidal-post-detection/tree/main/Flask

## Contents of this Section
* [Pre-Requisites for the section](#prerequisite)
* [Installing Libraries](#install)
* [Define File Structure](#file)
* [Create Prediction Pipeline](#pred)
* [Final Flask File](#flask)
* [Testing Application](#test)
* [Deploy Project on Heroku](#deploy)
* [Summary and Conclusions](#summary)

## <a name="prerequisite">Pre-Requisites for this section</a>
To implement the complete project you will need the following:
* Any operating system Linux, Windows, or Mac OS.
* Python 3+ installed https://www.python.org/
* Basic Python programming knowledge https://docs.python.org/3/tutorial/
* Basic Git knowledge https://git-scm.com/


## <a name="install">Step 1 - Installing Flask and related package</a>  
* Before installing Flask, we will create separate `python environment` for this project. If you are using Anaconda, open Anaconda Prompt, or else open your Command Prompt in Windows. Linux users can open their terminal instead. Enter the following code to create an environment named as `venv`.
```python
python3 -m venv env
```
* To use this environment we need to activate it using following code.
```python
source venv/bin/activate
```
* Environment is activated now we can download all packages required for the project.  
A few packages are requrired for this project like pytorch, numpy, pandas, transformers, and pickel.
```python
pip install flask pytorch torchvision numpy pandas transformer pickel
```

## <a name="file">Step 2 - File Structure</a>  
We need to have a file structure for best practice.
```js
─── Flask
    ├── ml_model
    │    ├── mlPipeline.py ------- (File where we define our MLRun pipeline)
    │    └── serving.py ----------- (File where we serve our pipeline for tranfomation, training, and testing)
    ├── file_upload (Folder where bulk CSV file stored)
    ├── templates (Folder with all html files)
    └── app.py -------------------- (Main Flask File)
```

## <a name="pred">Step 3 - Creating prediction pipeline.</a>
We define our pipeline script in **`mlPipeline.py`** file under **`ml_model`** folder.  
Final pipeline is as follows - 
    **Data -> Pre-processing -> Model -> Prediction -> Final Result**  

<p align="center">
  <img src="https://github.com/ahmadkhan242/Suicidal-post-detection/blob/main/images/mlRunFlow.png" />
</p>

## <a name="flask">Step 4 - Final Flask script.</a>
This final script is to be written in `app.py` file. This file will handel all HTTP requests we are going to use.    
In this file we will `import pred()` function we created in pipeline section then `import Flask` to create an app instance.
```python
    app = Flask(__name__)
```
<details> 
    <summary>Final code for <b>app.py</b> is file present here.</summary>
    <h3 style="display:inline-block"><summary>All this code to be written in <u><i>app.py</i></u> fie. </summary></h3>
    
```python
    # Importing pred function from ml_model/predict.py file.
    from ml_model.predict import pred

    import os
    from flask import Flask, render_template, request, make_response, jsonify, send_file
    
    # Create an app instance using Flask
    app = Flask(__name__)

    # Set up the main route
    @app.route('/predict', methods=['POST'])
    def home():
        print("Action Initiated")
        review = request.json['review']
        prediction = pred(review)
        print(prediction)
        return prediction

    if __name__ == '__main__':
        app.run()

```
    
</details>

We will use this instance to handle HTTP request. First we will define a route decorator `@app.route()` then pass API end name, here we use **`/predict`**.
Since, we will be receiving our text data on this route, here we will pass method as `POST`.  
Learn about basics of newtorking if you don't know [Learn Here](https://www.toolsqa.com/rest-assured/rest-routes/)
```python
    @app.route('/predict', methods=['POST'])
```
Now, we will define a fuction which will be executed when someone make POST request on `/predict` route. In this function we will take our POST route data and pass to the `pred()` function we imported from **`ml_model/predict.py`** file.
```python
    def home():
        review = request.json['review']
        prediction = pred(review)
        print(prediction)
        return prediction
```

## <a name="test">Step 4 - Testing final app.</a>
To run the Flask open terminal in the project directory and type following command in terminal.
```bash
    flask run
```
> Output
<p align="center">
    <kbd>
  <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/flask_run.png">
        </kbd>
</p>

* Now we can test our API on `http://127.0.0.1:5000/` with `/predict` endpoint we have created in `app.py` file.
* To test your final API you can use any API Client (Postman, Insomnia, etc).  

# <a name="deploy">Deployment on Heroku.</a>
We expect you have GitHub account and the knowledge of how to create repository. If not, [learn here](https://guides.github.com/activities/hello-world/)
1. **Create a file naming `Procfile` in main directory. This file specify which command to run at app startup, for our app write this command**
    ```bash
    web: gunicorn --bind 0.0.0.0:$PORT main_app:app
    ```

<details> 
<summary>Explanation with Screenshot</summary>  
    
`requirements.txt` file is used by the heroku server to download all packages. If we specify `torch == 1.6.0`, then it will download whole pytorch library. Since we are using free version of heroku, we have only `CPU` support not `GPU`. Pytorch library comes with all files required for GPU and CPU support, so we need to download only `CPU` specific files. As Heroku only provides `500 MB` storage in free version and complete Pytorch library is more than 600 Mb, we switch to the CPU version as it takes only 160 MB of space.
    <table>
        <tr>
        <th>Before Update</th>
        <th>After Update</th>
        </tr>
        <tr>
        <td>
        <p align="center">
          <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/before_req.png">
        </p> 
        </td>
        <td>
        <p align="center">
          <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/after_req.png">
        </p> 
        </td>
        </tr>
    </table>
</details>
    
2. **Now create a repository and push all code on github repository. If you don't know how to do that, learn it [here](https://www.datacamp.com/community/tutorials/git-push-pull)**
3. **Create Heroku account https://signup.heroku.com/ and Create New App.**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/h1.png">
                </kbd>
        </p>
    </details>

4. **Choose App name and region.**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/h2.png">
                </kbd>
        </p>
    </details>

5. **Link your Github account with Heroku, search your repository where you have pushed all your code and `connect`.**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/h3.png">
                </kbd>
        </p>
    </details>

6. **Choose branch, `Enable automatic deploy` so that it can automatically build your app when you push any changes to your repository and hit `Deploy Branch`.**
    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/h4.png">
                </kbd>
        </p>
    </details>

7. **You can see App Build log. It will display any errors if occurs.**

    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/h5.png">
                </kbd>
        </p>
    </details> 
    
8. **Finally after successful build you can launch your app by clicking `View`**
    * But in this case we have not include any frontend we will not see anything, although we can test our API in the same way we tested on Local by just replacing Localhost with the new URL.
        > http://127.0.0.1:5000/predict => http://NEW_URL/predict

    <details> 
        <summary>Detail Screenshot</summary>
        <p align="center">
            <kbd>
              <img src="https://github.com/ahmadkhan242/Transfer-Learning-Model-hosted-on-Heroku-using-React-Flask/blob/main/Images/heroku/h6.png">
                </kbd>
        </p>
    </details> 
    
   
### Refrence
* https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3
*** 

 
 

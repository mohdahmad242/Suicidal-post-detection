

# importing necssary libraries 
from ml_model.serving import pred, mlRunPipeline, bulktraining
import os
from flask import Flask, render_template, request, make_response, jsonify, send_file
import pandas as pd

# getting current working directory
cwd = os.getcwd()


# initializing flask app object
app = Flask(__name__, template_folder='templates')



# calling to MLRUN pipeline 
mlRunPipeline()


# app route for home page which has single input evaluation
@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'GET':
		return render_template('main.html', result={"result": False}) # in get method it will render the template and redirect to main.html
    
	if request.method == 'POST':
		review = request.form['url'] # take an input of text from the form used in html
		prediction = pred(review) # pred function will get called and it will predict from the serving
		prediction = {
            "text": review,
			'result': prediction
		}
		return render_template('main.html', result=prediction) # prediction output will be rendered on the UI template


	
# app route for bulktraining of data
@app.route('/bulktraining', methods=['POST', 'GET'])
def upload_file():
	if request.method == 'GET':
		return render_template('bulkTraining.html', result={"result": False})

	if request.method == 'POST':
		file = request.files['data_file'] # getting the bulk training csv
		data = pd.read_csv(file)
		print("DF - ",data.head())
		data.to_csv(cwd +"/file_upload/bulkTrain.csv", index=False) # storing the bulk training csv
		filepath = cwd + "/file_upload/bulkTrain.csv"
		success = bulktraining(filepath) # bultraining the whole data obtained
		return render_template('bulkTraining.html', result={"result": True}) #rendering thr UI template after successful training of the data

# main containing app.run() to run the flask app
if __name__ == '__main__':
    app.run()

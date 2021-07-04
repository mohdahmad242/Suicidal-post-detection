from ml_model.serving import pred, mlRunPipeline, bulktraining
import os
from flask import Flask, render_template, request, make_response, jsonify, send_file


app = Flask(__name__, template_folder='templates')

mlRunPipeline()

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'GET':
		return render_template('main.html', result={"result": False})
    
	if request.method == 'POST':
		review = request.form['url']
		prediction = pred(review)
		prediction = {
			'result': prediction
		}
		return render_template('main.html', result=prediction)


@app.route('/bulktraining', methods=['POST', 'GET'])
def upload_file():
	if request.method == 'GET':
		return render_template('bulkTraining.html', result={"result": False})


	if request.method == 'POST':
		file = request.files['file']
		data = pd.read_csv(file)
		print("DF - ",data.head())
		data.to_csv("file_upload/bulkTrain.csv", index=False)
		filepath = "file_upload/bulkTrain.csv"
		success = bulktraining(filepath)
		return render_template('bulkTraining.html', result={"result": True})


if __name__ == '__main__':
    app.run()

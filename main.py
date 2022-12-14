from flask import Flask, jsonify, request, redirect
import pandas as pd
from flask_cors import CORS
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from werkzeug.utils import secure_filename
import urllib.request
import os
import calendar

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "C:/Users/Developer/Documents/Daposta/study/analytics-project"
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 102
filename1 = ''

@app.route("/")
def hello():
  return "Hello World!"


@app.route("/sales-by-product")
def sales_by_product():
    df = pd.read_csv('sales.csv', usecols=["item", "sales"])
    df = df.groupby('item', as_index=False)['sales'].sum()
    result = df.to_dict('records')
    data = [{'name':x['item'],'sales': x['sales'] } for x in result]
    return  jsonify({'data': data})


@app.route("/sales-by-year")
def sales_by_year():
    df = pd.read_csv('sales.csv', usecols=["date", "sales"])
    df["date"] = pd.to_datetime(df["date"])
    df['year'] = df['date'].dt.year
    df = df.groupby('year', as_index=False)['sales'].sum()
    result = df.to_dict('records')
    # data = [{'year':x['year'],'sales': x['sales'] } for x in result]
    year_data =  [x['year'] for x in result]
    sales_data =  [x['sales']  for x in result]
    return jsonify({'data': {
        'years': year_data, 'sales':sales_data
    }})



@app.route("/sales-prediction")
def sales_prediction():
    model = pickle.load(open('model.pkl','rb'))
    # df = pd.read_csv('sales.csv', usecols=["quantity","discount","unit_price", "month", "sales"])
    #send data in a loop from month 1 to month with start date and end date for each month
    year = 2023
    month_items =  [x for x in range(1, 13)] #, key=lambda x:x[0])
    months_in_year = [] 
    for month in month_items:
        res = calendar.monthrange(year, month)
        # month = res[0]
        day = res[1]
        months_in_year.append((month, day) )
    
    predictions = []
    for i in months_in_year:
        monthly_prediction = {}
        month = str(i[0]) if len(str(i[0])) > 1 else str('0'+ str(i[0]))
        last_day = i[1]
        print(month, last_day)
        index_future_dates = pd.date_range(start=f'{str(year)}-{month}-1', end=f'{str(year)}-{month}-{str(last_day)}')
        prediction = model.predict(start=1, end=last_day, type='levels').rename('ARIMA Predictions')
        print(index_future_dates)
        result = round(prediction[-1], 2)
        print(result)
        end= f'{str(year)}-{month}-{str(last_day)}'
        predictions.append({ end: result})

    
    print(predictions)
    return jsonify({'data':predictions})

ALLOWED_EXTENSIONS = set(['csv', ])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/file-upload', methods=['POST'])
# def upload_file():
# 	# check if the post request has the file part
# 	if 'file' not in request.files:
# 		resp = jsonify({'message' : 'No file part in the request'})
# 		resp.status_code = 400
# 		return resp
# 	file = request.files['file']
# 	if file.filename == '':
# 		resp = jsonify({'message' : 'No file selected for uploading'})
# 		resp.status_code = 400
# 		return resp
# 	if file and allowed_file(file.filename):
# 		filename = secure_filename(file.filename)
# 		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         filename1 = filename
#         print('filename1  >>', filename1)
#         resp = jsonify({'message' : 'File successfully uploaded'})
# 		resp.status_code = 201
# 		return resp
# 	else:
# 		resp = jsonify({'message' : 'Allowed file types is csv'})
# 		resp.status_code = 400
# 		return resp



if __name__ == "__main__":
  app.run()
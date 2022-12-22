from flask import Flask, jsonify, request, redirect
import pandas as pd
from flask_cors import CORS, cross_origin
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

UPLOAD_FOLDER = os.getcwd() #"C:/Users/Developer/Documents/Daposta/study/analytics-project"
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 102
filename1 = ''

@app.route("/")
def hello():
  return "Hello World!"


@app.route("/sales-by-product")
def sales_by_product():
    # df = pd.read_csv('sales.csv', usecols=["item", "sales"])
    # global filename1
    print('filename>.>', filename1)
    df = pd.read_csv(filename1, usecols=["item", "sales"])
    df = df.groupby('item', as_index=False)['sales'].sum()
    result = df.to_dict('records')
    data = [{'name':x['item'],'sales': x['sales'] } for x in result]
    return  jsonify({'data': data})


@app.route("/sales-by-year", methods = ['GET'])
@cross_origin(origin='*')
def sales_by_year():
    # df = pd.read_csv('sales.csv', usecols=["date", "sales"])
    # global filename1
    df = pd.read_csv(filename1, usecols=["date", "sales"])
    
    
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
# @cross_origin(origin='*')
def sales_prediction():
    model = pickle.load(open('model.pkl','rb'))
    df = pd.read_csv('sales.csv', usecols=["quantity","discount","unit_price", "month", "sales"])
    # df = pd.read_csv(filename1, usecols=["quantity","discount","unit_price", "month", "sales"])
    #send data in a loop from month 1 to month with start date and end date for each month
    year = 2023
    month_items =  [x for x in range(1, 13)] #, key=lambda x:x[0])
    months_in_year = [] 
    month_data = []
    prediction_data = []
    for month in month_items:
        res = calendar.monthrange(year, month)
        # month = res[0]
        day = res[1]
        months_in_year.append((month, day) )

    for i in months_in_year:
        monthly_prediction = {}
        month = str(i[0]) if len(str(i[0])) > 1 else str('0'+ str(i[0]))
        last_day = i[1]
        # index_future_dates = pd.date_range(start=f'{str(year)}-{month}-1', end=f'{str(year)}-{month}-{str(last_day)}')
        prediction = model.predict( start=1, end=last_day, type='levels').rename('ARIMA Predictions')
        # start_date = f'{str(year)}-{month}-1'
        # end_date = f'{str(year)}-{month}-{str(last_day)}'
        # print(start_date, end_date)
        # prediction = model.predict(df, start=start_date, end=end_date, type='levels').rename('ARIMA Predictions')
        result = round(prediction[-1], 2)
        # end= f'{str(year)}-{month}-{str(last_day)}'
        month_name = convert_month(month)

        prediction_data.append({'month': month_name, 'sales':result})
    
    return  jsonify({'data':prediction_data})

ALLOWED_EXTENSIONS = set(['csv', ])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    # check if headers we need exist
    # df = pd.read_csv(request.files['file'])
    # list_of_column_names = list(df.columns)
    # if not 'date' and 'sales' and 'item' in list_of_column_names:
    #     resp = jsonify({'message' : 'Required headers sales/item/date are missing'})
    #     resp.status_code = 400
    #     return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        df = pd.read_csv(filename)
        list_of_column_names = list(df.columns)
        print('list_of_column_names>>', list_of_column_names)
        if not ('date' and 'sales' and 'item') in list_of_column_names:
            resp = jsonify({'message' : 'Required headers sales/item/date are missing'})
            resp.status_code = 400
            return resp
        global  filename1 
        filename1= filename
        # print('filename1  >>', filename1)
        resp = jsonify({'message' : 'File successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify({'message' : 'Allowed file types is csv'})
        resp.status_code = 400
        return resp


def convert_month(month_num):
    match month_num:
        case '01':
            return 'January'
        case '02':
            return 'February'
        case '03':
            return 'March'
        case '04':
            return 'April'
        case '05':
            return  'May'
        case '06':
            return 'June'
        case '07':
            return 'July'
        case '08':
            return 'August'

        case '09':
            return 'September'
        case '10':
            return 'October'
        case '11':
            return 'November'
        case '12':
            return'December'
        case _:
            return ''






@app.route("/sales-prediction-chart")
# @cross_origin(origin='*')
def sales_prediction_chart():
    model = pickle.load(open('model.pkl','rb'))
    df = pd.read_csv('sales.csv', usecols=["quantity","discount","unit_price", "month", "sales"])
    # df = pd.read_csv(filename1, usecols=["quantity","discount","unit_price", "month", "sales"])
    #send data in a loop from month 1 to month with start date and end date for each month
    year = 2023
    month_items =  [x for x in range(1, 13)] #, key=lambda x:x[0])
    months_in_year = [] 
    month_data = []
    prediction_data = []
    for month in month_items:
        res = calendar.monthrange(year, month)
        # month = res[0]
        day = res[1]
        months_in_year.append((month, day) )

    for i in months_in_year:
        monthly_prediction = {}
        month = str(i[0]) if len(str(i[0])) > 1 else str('0'+ str(i[0]))
        last_day = i[1]
        # index_future_dates = pd.date_range(start=f'{str(year)}-{month}-1', end=f'{str(year)}-{month}-{str(last_day)}')
        prediction = model.predict( start=1, end=last_day, type='levels').rename('ARIMA Predictions')
        # start_date = f'{str(year)}-{month}-1'
        # end_date = f'{str(year)}-{month}-{str(last_day)}'
        # print(start_date, end_date)
        # prediction = model.predict(df, start=start_date, end=end_date, type='levels').rename('ARIMA Predictions')
        result = round(prediction[-1], 2)
        # end= f'{str(year)}-{month}-{str(last_day)}'
        month_name = convert_month(month)

        prediction_data.append({'month': month_name, 'sales':result})
    
    return  jsonify({'data':prediction_data})

if __name__ == "__main__":
  app.run()
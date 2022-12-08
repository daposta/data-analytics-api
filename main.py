from flask import Flask, jsonify
import pandas as pd
data = None
app = Flask(__name__)

@app.route("/")
def hello():
  return "Hello World!"


@app.route("/sales-by-product")
def sales_by_product():
    df = pd.read_csv('sales.csv', usecols=["item", "sales"])
    # df["date"] = pd.to_datetime(df["date"])
    # df['year'] = df['date'].dt.year
    df = df.groupby('item', as_index=False)['sales'].sum()
    result = df.to_dict('records')
    data = [{'name':x['item'],'sales': x['sales'] } for x in result]
    return jsonify({'data': data})


@app.route("/sales-by-year")
def sales_by_year():
    df = pd.read_csv('sales.csv', usecols=["date", "sales"])
    df["date"] = pd.to_datetime(df["date"])
    df['year'] = df['date'].dt.year
    df = df.groupby('year', as_index=False)['sales'].sum()
    result = df.to_dict('records')
    data = [{'year':x['year'],'sales': x['sales'] } for x in result]
    return jsonify({'data': data})

if __name__ == "__main__":
  app.run()
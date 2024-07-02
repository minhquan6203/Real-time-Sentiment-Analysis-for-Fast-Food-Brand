from flask import Flask, render_template, jsonify
from bson import json_util
import json
from pymongo import MongoClient

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client.fb_db
collection = db.fb_collection

@app.route('/')
def kfc():
    return render_template('dashboard.html', entity="KFC", title="KFC Dashboard")

@app.route('/subway')
def subway():
    return render_template('dashboard.html', entity="Subway", title="Subway Dashboard")

@app.route('/dominos_pizza')
def dominos_pizza():
    return render_template('dashboard.html', entity="DominosPizza", title="Dominos Pizza Dashboard")

@app.route('/mcdonalds')
def mcdonalds():
    return render_template('dashboard.html', entity="McDonalds", title="McDonalds Dashboard")

@app.route('/data/<entity>')
def get_data(entity):
    # Tìm những dữ liệu có Entity tương ứng
    data = list(collection.find({"Entity": entity}))
    # Convert ObjectId to string
    for item in data:
        item['_id'] = str(item['_id'])
    # Serialize data to JSON
    return json.dumps(data, default=json_util.default)

if __name__ == '__main__':
    app.run(debug=True)
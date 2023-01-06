import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app =Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))
@app.route('/home')
def home():
    #return "Hellow worls"
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']
    print(data)
    new_data=[list(data.values())]
    output = model.predict(new_data)[0]
    return jsonfy(output)

app.route('/predict', methods=['POST'])
def predict():

    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
   
    output = model.predict(final_features)[0]
    print(output[0])
    return render_template('home.html', prediction_text = "Airfoil pressure is{}".format(output[0]))

if __name__ =="__main__":
    app.run(debug=True)
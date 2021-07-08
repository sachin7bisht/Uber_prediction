import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import math

app=Flask(__name__)

model=pickle.load(open('taxi.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    fet=[int(x) for x in request.form.values()]
    final_feat=[np.array(fet)]
    predictn=model.predict(final_feat)
    output=round(predictn[0],2)
    return render_template('index.html',prediction_text='Number of rides {}'.format(math.floor(output)))

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080)



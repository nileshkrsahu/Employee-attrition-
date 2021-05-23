import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from load import init
app=Flask(__name__)
#global model

#model= init()

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    d=pd.DataFrame()
    d=d.append(final_features)
    d.to_csv('f.csv')
    final = np.vstack((final_features,final_features))
    
    predict_out=init(final)
    print("Value_Predicted ")
    return render_template('index1.html',prediction_text='Employee Attrition $ {}'.format(predict_out[0]))
    
if __name__ == "__main__":
    app.run(debug=True)
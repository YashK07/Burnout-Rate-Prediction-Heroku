from flask import Flask,render_template
import os
from flask import request
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

#from xgboost import XGBRegressor

app = Flask(__name__) #Initialize the flask app
model = pickle.load(open("model\\bunrout_model_xgb.pkl", 'rb'))


@app.route('/')
def home():
    return render_template('index_1.html')

@app.route('/predict',methods=['POST'])
def predict():
    if(request.method=="POST"):
        int_feat = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'Gender', 'Company Type', 'WFH Setup Available']
        l = []
        for i in int_feat:
            val = int(request.form[i])
            l.append(val)

    """
    print("***")
    req_data = request.get_json(force = True)
    print(req_data)
    int_feat = ['designation', 'resource', 'mfs', 'gender', 'company', 'wfh']
    l = []
    for i in int_feat:
        print(int(req_data[i]))
        l.append(int(req_data[i]))
    print(l)

    """
        #convert into array of shape -> (1,6)
    feat_arr = np.array(l).reshape(-1,1).reshape(1,6)
    input = pd.DataFrame(feat_arr,columns = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'Gender_Female', 'Company Type_Service', 'WFH Setup Available_Yes'])
    prediction = float(model.predict(input)[0])

    prediction = round(prediction, 2)
    stat = 0
    #top 25 percentile
    if(prediction<=0.3):
        feedback1 = "Fantastic! You have a low burnout rate of {} .".format(prediction)
        return render_template("index_1.html",color = "color:#33CC00;",feedback = feedback1)
    #top 25 percentile to 75 percentile
    elif((prediction>0.3) & (prediction<=0.59)):
        feedback2 = "Not bad...You have a moderate burnout rate of {} .".format(prediction)
        return render_template("index_1.html",color = "color:#339900;",feedback = feedback2)
    #top 75 percentile to 90 percentile
    elif((prediction>0.59) & (prediction<=0.80)):
        feedback3 = "Oops!! You have a high burnout rate of {} .".format(prediction)
        return render_template("index_1.html",color = "color:#FF0000;",feedback = feedback3)
    #top 90 percentile to 99 percentile
    elif((prediction>0.8) & (prediction<=0.9)):
        feedback4 = "Ouch!!! You have a very high burnout rate of {} .".format(prediction)
        return render_template("index_1.html",color ="color:#CC0000;",feedback = feedback4)
    #top 99 percentile
    else:
        feedback5 = "Sorry! You have an extremly high burnout rate of {} .".format(prediction)
        return render_template("index_1.html",color ="color:#990000;",feedback = feedback5)
    """
    else:
        return render_template('index.html',feedback = "Your burnout rate will show up here.")
    """


if __name__ == "__main__":
    app.run('debug'==True)

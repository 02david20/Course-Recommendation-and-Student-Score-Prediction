from flask import Flask, request, render_template
import numpy as np
import pickle
from utils.cfUtil import *

app = Flask(__name__)
# Import ML model
loaded_model = pickle.load(open("model/model.pkl", "rb"))
final_data_matrix = pickle.load(open("model/final_data_matrix.pkl", "rb"))


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 3822)
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/')
def hello():
    return render_template("welcome.html", title='Home')

@app.route('/predictor')
def predictor():
    return render_template("predictor.html", title='Predictor')

 
@app.route('/predictor/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        name = to_predict_list['name']
        fname = to_predict_list['fname']
        del to_predict_list['name']
        del to_predict_list['fname']

        # Feature extraction
        ks = [key for key in to_predict_list]
        category_name = [to_predict_list[k] for k in ks[:4]]
        scores_features = [float(to_predict_list[k]) for k in ks[4:]]

        with open('feature.txt', 'r') as f:
            categories = f.read()
            categories = categories.split(',')
            del categories[-8:-1]

        category_features = []
        for category in categories:
            if category in category_name:
                category_features.append(1)
            else:   
                category_features.append(0)
        features = category_features+scores_features
                
        result = ValuePredictor(features)       
        if int(result)== 1:
            prediction ='Congratulation. {} {} có vẻ như sẽ qua môn'.format(fname,name)
        else:
            prediction ='{} {} nên chăm chỉ hơn'.format(fname,name)        
        return render_template("result.html", prediction = prediction)

@app.route('/cf')
def cf():
    return render_template("cf.html")

@app.route('/cf/result', methods = ['POST'])
def cf_res():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        for k,v in to_predict_list.items():
            to_predict_list[k] = float(v)
        result = calculate_from_input(to_predict_list,final_data_matrix, 'user')
        return render_template("cf_result.html", result=result)



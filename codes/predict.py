"""
Create on 7 Nov,2020

@author:nannan.sun@wowjoy.cn

1.预测TOC、IP微服务
"""
import  pandas as pd
import json
from sklearn.externals import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Predict_result(object):
    def __init__(self):
        self.columns = ["反应温度","氧气分压","反应时间","催化剂投加量","初试pH"]
        self.TOC_pred = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"model/TOC_Pred.pkl"))
        self.IP_pred = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"model/IP_Pred.pkl"))
    def predict_result(self,input_data):
        trans_dict = {"反应温度":[],"氧气分压":[],"反应时间":[],"催化剂投加量":[],"初试pH":[]}
        for name in self.columns:
            trans_dict[name].append(input_data[name])
        input_df = pd.DataFrame(trans_dict)
        print("-----input----",input_data)
        TOC_pred = self.TOC_pred.predict(input_df).ravel()
        IP_pred = self.IP_pred.predict(input_df).ravel()
        #result = json.load(json.dump(y_pred_result))
        Res = {"TOC":TOC_pred[0],"IP":IP_pred[0]}


        return Res

if __name__ == "__main__":
    
    input_dict = {"反应温度":220,"氧气分压":1.1,"反应时间":60,"催化剂投加量":5,"初试pH":4}
    Predict_result = Predict_result()
    y_pred = Predict_result.predict_result(input_dict)
    print(y_pred)

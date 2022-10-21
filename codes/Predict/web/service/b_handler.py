#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 15,2019

@author: Nannan.sun@wowjoy.cn

"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../")
import json
import tornado.web

from app.ie.modules import B_predicted

Get_result = B_predicted.Predict_result()

class B_Handler(tornado.web.RequestHandler):

    def get(self):
        '''
        @summary: 
        '''
        in_sent = self.get_argument('in_sent')
        print("in_sent:", in_sent)
        pre_result = Get_result.predict_result(in_sent)
        #import IPython
        #IPython.embed()
  
        print(json.dumps(json.loads(json.dumps(pre_result)),ensure_ascii=False))
        self.write(json.dumps(pre_result))
        
        #return rets

    def post(self):
        '''
         @summary: 对患者NCP阳性，阴性进行分类
        '''
       	in_json = self.request.body
        print("in_json", in_json)
        jsonstr = in_json.decode('utf8')
        print('json_字符串：', jsonstr)
        json_data = json.loads(jsonstr)
        print("json_data", json_data)

        pre_result = Get_result.predict_result(json_data)
        print(json.dumps(json.loads(json.dumps(pre_result)), ensure_ascii=False))
        self.write(json.dumps(pre_result))

    def put(self):

        '''
        @summary: 对患者NCP阳性，阴性进行分类
        '''
        pass


if __name__ == "__main__":

    input_dict = {"反应温度":220,"氧气分压":1.1,"反应时间":60,"催化剂投加量":5,"初试pH":4}
    #input_Data = pd.DataFrame(input_dict)
    #print(input_dict)
    #input_Data = pd.read_excel(input_Data_path)
    #Predict_result = Get_result()
    y_pred = Get_result.predict_result(input_dict)
    print(y_pred)




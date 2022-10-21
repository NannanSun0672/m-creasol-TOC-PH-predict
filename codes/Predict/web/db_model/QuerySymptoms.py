# -*- coding: utf-8 -*-

'''
Created on 2017年09月04日

@author: xiaoliang.qian@wowjoy.cn
'''

from sqlalchemy import Column, String, Integer, VARCHAR,ForeignKey, Float, DateTime 
from sqlalchemy.orm import relationship,backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from .BaseModel import BaseModel
#import datetime
#from settings import engine


class QuerySymptoms(BaseModel):
    __tablename__ = 'query_symptoms'
    id = Column(Integer, primary_key=True, autoincrement=True)
    chief_complaint = Column(VARCHAR(500))  # 主诉
    pred_label = Column(VARCHAR(32), default="")    # 预测的label
    depart_label = Column(VARCHAR(32), default="")  # 真实的label
    query_time = Column(DateTime, server_default=func.now())   # 时间

    def __repr__(self):
        return "<QuerySymptoms(cc='%s', label='%s')>" % (self.chief_complaint, self.depart_label)

#BaseModel.metadata.create_all(engine)


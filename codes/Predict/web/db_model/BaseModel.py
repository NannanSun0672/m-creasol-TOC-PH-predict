# -*- coding: utf-8 -*-

'''
Created on 2017年09月04日

@author: xiaoliang.qian@wowjoy.cn
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../")
from conf.settings import MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_HOST, MYSQL_DB_NAME

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine('mysql+pymysql://%s:%s@%s/%s?charset=utf8' % 
                       (MYSQL_DB_USER, MYSQL_DB_PWD, MYSQL_DB_HOST, MYSQL_DB_NAME),
                       encoding='utf-8', echo=False, pool_size=100, pool_recycle=10)
DB_Session = sessionmaker(bind=engine)
session = DB_Session()

BaseModel = declarative_base()


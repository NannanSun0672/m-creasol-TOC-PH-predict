# -*- coding: utf-8 -*-

'''
Created on 2018年08月27日

@author: Nannan.sun@wow.cn
'''
#from web.service.ner_handler import NERHandler
from web.service.a_handler import A_Handler
from web.service.b_handler import B_Handler


url_patterns=[
              #(r"/ner", NERHandler),
             (r"/A_module",A_Handler),
             (r"/B_module",B_Handler)

             ]


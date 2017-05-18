# -*- coding: utf-8 -*-
"""
神经网络设计
3.2.4 Hopfield网络
单个satlins函数作为递归（反馈）层
输入向量，生成标准模式向量

Created on Thu May 18 18:39:15 2017

@author: yf
"""
import numpy as nplib

#===========================================
# 以下为 Hopfield 网络的封装实现
#===========================================
class Hopfield3VectorNN:
    __W = nplib.array([[0.2, 0, 0], [0, 1.2, 0], [0, 0, 0.2]], dtype=nplib.float)
    __recurTimes = 100
    __b = nplib.array([0.9, 0, -0.9])
    
#    def __init__(self):
#        self.__W1 = weight1
#        self.__b1 = nplib.array([weight1[0,:].size, weight1[0,:].size], dtype=nplib.float)
        
    def __recurrentSatlin(self, a0, times):
        aNext = self.__W.dot(a0)+self.__b
        nplib.clip(aNext, -1.0, 1.0, out=aNext)
        
        print("__recurrentSatlin", aNext, " ", times )
        
        times-=1;
        if times>0:
            aNext = self.__recurrentSatlin(aNext,times)
         
        return aNext
    #---------------------------------Neural Response    
    def response(self, p):      
        return self.__recurrentSatlin(p, self.__recurTimes)
#===========================================
    
xx = Hopfield3VectorNN();
print("[-1, -1, -1] Hopfield response:", xx.response(nplib.array([-1,-1,-1], dtype=nplib.float) ) )


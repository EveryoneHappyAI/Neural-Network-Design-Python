# -*- coding: utf-8 -*-
"""
神经网络设计
3.2.3 Hamming网络
purelin函数作为前馈层，poslin函数作为递归（反馈）层
专门为求解二值模式识别问题设计

Created on Thu May 18 01:13:49 2017

@author: 7up
"""

import numpy as nplib

#输入
p=nplib.array([-1,-1,-1])

#前馈
#权值
W1=nplib.array([[1,-1,-1],[1,1,-1]])
#偏置
b1=nplib.array([3,3])

a1=W1.dot(p)+b1
           
#递归反馈层 
W2=nplib.array([[1,-0.5],[-0.5,1]])

#递归Poslin函数（人工控制递归次数times）
def RecurrentPoslin(w,a,times=1):    
    aNext = w.dot(a);
    nplib.clip(aNext, 0, 100, out=aNext)
    times-=1;
    if times>0:
        aNext = RecurrentPoslin(w,aNext,times)
            
    return aNext

print(W1.dot(p))
print(a1)
print(RecurrentPoslin(W2,a1,1))
# 以上为书中实例、

# 以下为Hamming网络的封装实现
#===========================================
class HammingNN:
    __W1 = nplib.array    
    __W2 = nplib.array([[1,-0.5],[-0.5,1]])
    __recurTimes = 1
    __b1 = nplib.array
    
    def __init__(self, weight1, weight2):
        self.__W1 = weight1
        self.__W2 = weight2
        self.__b1 = nplib.array([weight1[0,:].size, weight1[0,:].size])
        
    def __preNerual(self, p, out=None):
        out = self.__W1.dot(p) + self.__b1
        return out
        
    def __recurrentPoslin(self, a0, times):
        aNext = self.__W2.dot(a0);
        nplib.clip(aNext, 0, 100, out=aNext)
        times-=1;
        if times>0:
            aNext = RecurrentPoslin(aNext,times)
         
        return aNext
    #---------------------------------Neural Response    
    def response(self, p):
        return self.__recurrentPoslin(self.__preNerual(p), self.__recurTimes)      
#===========================================
    
xx = HammingNN(nplib.array([[1,-1,-1],[1,1,-1]]), nplib.array([[1,-0.5],[-0.5,1]]));
print("[-1,1,-1] Hamming response:", xx.response([-1,1,-1]) )



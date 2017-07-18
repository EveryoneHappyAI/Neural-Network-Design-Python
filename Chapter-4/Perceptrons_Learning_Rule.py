# -*- coding: utf-8 -*-
"""
神经网络设计
4.2.3 感知机学习规则
基本的感知机hardlim 或者 hardlims函数的感知机
通过训练，寻找合适的权值以及偏置值
该网络可以完成简单的二值 学习、以及分类

Created on Fri May 19 13:54:38 2017

@author: yf
"""

import numpy as nplib
import matplotlib.pyplot as plt 

#===========================================
# 以下为感知机学习规则的封装实现
#===========================================
class PerceptronRuleNN:
    __W = nplib.array([1,1])
    __WLast = nplib.array([])
    
    __b = 0.5 #nplib.array
    __bInit = nplib.array
    
    #def __init__(self):
        #self.__W1 = weight
        #self.__b1 = nplib.array([weight1[0,:].size, weight1[0,:].size], dtype=nplib.float)
        
#    def __recurrentPoslin(self, a0, times):
#        aNext = self.__W2.dot(a0)
#        nplib.clip(aNext, 0.0, 100.0, out=aNext)
#        
#        print("__recurrentPoslin", aNext, " ", times )
#        
#        times-=1;
#        #if times>0:
#        if (a0-aNext).dot(nplib.array([1,1])) != 0 and times>0 :
#            aNext = self.__recurrentPoslin(aNext,times)
#         
#        return aNext
    
    def __hardlim(self, inValue):
        if (inValue<0):
            return 0
        else:
            return 1
        
    #---------------------------------Neural Response
    def response(self, p):
        return self.__hardlim( (self.__W.dot(p)+self.__b))
    
    #---------------------------------Neural Tranning
    # pArray 训练数据输入队列， tArray正确结果队列
    def train(self, pArray, tArray):
        if(pArray.ndim<=1):
            print("Trainning data not correct!")
            return
        #initialize weight in range (0,1]
        self.__w = nplib.random.ranf(pArray[0].shape)
        
        print("*****", self.__w)
        
        #adjust the weight and b. By the right result
        for i in range(pArray.shape[0]):
            a = self.__hardlim((self.__W.dot(pArray[i]) + self.__b))
            print(" =======  ", self.__w, " ======== ", i, "  =============  " , a)
            e = tArray[i]-a
            self.__w = self.__w + e*pArray[i]
            self.__b = self.__b + e
        
        print("Weight and b after trainning... W: ", self.__w, "  b: ", self.__b)
    
#===========================================
 

#============测试过程   

xx = PerceptronRuleNN();

#训练数据 aa
aa = nplib.array([[1,-1], [2,-1]])#, nplib.zeros((2,3)), nplib.zeros((2,3)), nplib.zeros((2,3))], dtype=nplib.float);

#训练
xx.train(aa,[1,1])

#print("---- ", aa, aa.shape, aa.dtype, aa.data, aa.flags, aa.flat, aa.imag, aa.real, aa.size, aa.itemsize, aa.nbytes, aa.ndim )
print("---- ", aa)
print(" shape: ", aa.shape)
print(" dtype: ", aa.dtype)
print(" data: ", aa.data)
print(" flags: ", aa.flags)
print(" flat: ", aa.flat)
print(" imag: ", aa.imag)
print(" real: ", aa.real)
print(" size: ", aa.size)
print(" itemsize: ", aa.itemsize)
print(" nbytes: ", aa.nbytes)
print(" ndim: ", aa.ndim)

#plt.figure()
plt.plot(aa[:,0],aa[0,:],'o')
#plot(aa)
help(plt)

#实际响应 xx.response()

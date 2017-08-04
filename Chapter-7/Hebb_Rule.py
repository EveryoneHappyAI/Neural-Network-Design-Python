# -*- coding: utf-8 -*-
"""
神经网络设计
7.*.* 有监督的Hebb学习
简单的自联想存储器模型
Hebb 假设：当细胞A的轴突到细胞B的距离近到足够激励它，且反复或持续的刺激B，那么在这两个细胞或一个细胞中将发生某种增长过程或代谢反应，增加A对细胞B的刺激效果

自联想存储器的简单模型，以及用它来识别数字 

Created on Fri Aug  4 19:12:13 2017

@author: yf
"""


import numpy as nplib
import matplotlib.pyplot as plt 

import cv2

#===========================================
# 以下为Hebb自联想网络
#===========================================
class HebbSelfAssociatNN:
    __W = None
    
    __b = 0 #nplib.array
    
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
    
    def hardlim(self, inValue):
        inValue[inValue<=0] = -1
        inValue[inValue>0] = 1
#        nplib.where(ou <0, ou, 0)
#        nplib.where(ou >0, ou, 1)
        return inValue
        
    #---------------------------------Neural Response
    def response(self, p):
        return self.hardlim( (self.__W.dot(p)+self.__b))
    
    #---------------------------------Neural Tranning
    # pArray 训练数据输入队列， tArray正确结果队列
    def train(self, pArray):
        if(pArray.ndim<=1):
            print("Trainning data not correct!")
            return
        #initialize weight in range (0,1]        
        
        p = pArray[0]
        self.__W = p.dot(p.T)
        print("*****", self.__W)
        self.__W = self.__W-self.__W
        print("*****", self.__W)
        #adjust the weight and b. By the right result
        for i in range(pArray.shape[0]):
            p = pArray[i]
            self.__W = self.__W + p.dot(p.T)
            print(" =======  ", self.__W, " ======== ", i, "  =============  " , p)
        
        print("Weight and b after trainning... W: ", self.__W)
    
#===========================================

nn = HebbSelfAssociatNN()

#arr = nplib.random.randn(10,6,5)


arr = nplib.arange(360).reshape(3,12,10)
print(arr)

for i in range(1,4):
    print('./'+ str(i) + '.bmp')
    
    img = cv2.imread('./'+ str(i) + '.bmp', 0)
    arr[i-1,:,:] = img #cv2.imread('./'+ str(i) + '.bmp', 0)
    
    cv2.imshow("testWindow", img)
    cv2.waitKey(-1)
    print(arr[i-1])
    
nn.hardlim(arr)

nn.train(arr)
outImg = nn.response(arr[0])

outImg = 255 * outImg
outImg[outImg<0] = 0
print(outImg)


cv2.imshow("testWindow", arr[1])
cv2.waitKey(-1)
cv2.destroyAllWindows()

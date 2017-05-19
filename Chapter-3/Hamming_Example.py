
# -*- coding: utf-8 -*-
"""
神经网络设计
3.2.3 Hamming网络 
purelin函数作为前馈层，poslin函数作为递归（反馈）层
专门为求解二值模式识别问题设计

Created on Thu May 18 01:13:49 2017

@author: 7up
"""

#书中的示例
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
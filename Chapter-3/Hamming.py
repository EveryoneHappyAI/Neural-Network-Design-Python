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
           
#          
           
print(W1.dot(p))
print(a1)



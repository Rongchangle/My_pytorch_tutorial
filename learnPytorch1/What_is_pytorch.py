'''
这部分相当于hello world了

'''

from __future__ import print_function
import torch
import numpy as np
'''
#1. tensor的宣布和基本运算
x = torch.Tensor(5, 3)
print(x)

x2 = torch.Tensor(5, 3)
torch.add(x,x,out = x2)
x4 = torch.add(x2,x2)
print(x2)
print(x4)

x21 = x + x + x
print(x2)


#2. tensor基本初始化,访问某个值,查询tensor大小
a = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
print(a[0,1]) #访问某个值,输出2
print(a[1]) #访问,输出好像是2,3,4,5,6
print(a.size()) #输出的是python的tuple

a = torch.randn(5,3)
a = torch.zeros(5, 3)
a = torch.ones(5, 3)
a2 = torch.ones(5, 3)



#3. tensor和numpy的相互转换
b = a.numpy()  #tensor转换成numpy
print(type(b))

#小心下面的加法,貌似b也会跟着一起变
a.add_(100)
print(a)
print(b)

#但是下面这种加法,b好像不会变,占时不明白,不过要避免可以考虑 a = a.numpy
a = a + 100
print(a)
print(b)

b = np.ones([5,3])
a = torch.from_numpy(b) #numpy转换成tensor(Torch张量)
print(a)

'''



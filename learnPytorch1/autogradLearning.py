'''
这部分难度大一点,关于autograd:自动分化的

pytorch神经网络的中心是autograd包,然后什么是自动分化??
autograd.Varialbe是中心类,它包括tensor,并且几乎支持tensor所有操作
完成计算调用.backward() 自动计算所有梯度
.data可以访问所有raw tensor  .grad和梯度有关
'''

import torch
from torch.autograd import Variable

#1.关于Variable宣布, grad_fn(创造者),backward,grad(求导)
x = Variable(torch.ones(2), requires_grad = True)

y = x + 2

print(x.grad_fn) #x是用户自己宣布的Variable,返回none
print(y.grad_fn) #y不是,返回一个我现在还看不懂的东西

z = y * y * 3
out = z.mean() #取值平均

gradients = torch.Tensor([1, 0.1])
out.backward(gradients) #求导数
print(x.grad) #输出xi对应导数和对应梯度的乘积
#忘记如何求导可以查看 https://ptorch.com/docs/3/autograd_tutorial







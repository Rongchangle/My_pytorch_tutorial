import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#用class定义一个神经网络
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) #很像tf的reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = F.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension, [1:]是切片语法,表示下标1到末尾
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()


#打印网络信息
print(net)
#打印网络中各个变量信息
params = list(net.parameters())
print(len(params))         #变量类别数,这里是10.因为卷积层2个,全连接层3个,所以(2+3)*2 = 10,(每个层都贡献w和b,所以要乘以2)
print(params[0].size())    #第一类变量shape(卷积层1的w)
print(params[1])           #打印第二类变量本身的值

#或者像下面那样遍历
for f in net.parameters():
    print(f.size())


#输入一幅'图片'到网络,得到输出
input = Variable(torch.ones(1,1,32,32)) #假造一张图片
output= net(input)  #网络执行
print(output)  #输出结果(就是图片属于X类的概率)


#定义损失函数,求导,输出各个参数梯度
target = Variable(torch.arange(1,11))
criterion = nn.MSELoss()
loss = criterion(output, target) # MSE是均方根误差,等价于loss = torch.mean((target-output)*(target-output))


#对损失函数求导,得到net中各个可训练变量的导数
net.zero_grad() #清除net中所有变量的梯度
print('before backward: ')
print(net.conv1.bias.grad)
loss.backward() #求导
print('after backward: ')
print(net.conv1.bias.grad) #输出对应导数
print(net.conv1.weight.grad)


#如何调整权重(上面的方法不过是输出对应导数而已,事实上不可能我们手动根据导数一个一个更新权重)
#下面是用optim包训练
optimizer = optim.SGD(net.parameters(),lr = 0.01) #宣布下降方法,学习率

#以下代码其实存在于training loop
#begin:
optimizer.zero_grad() #重置0,防止Variable梯度叠加
output = net(input) #动态建立图,据说可以改变图结构
loss = criterion(output,target)
loss.backward() #
optimizer.step() #更新权值
#end








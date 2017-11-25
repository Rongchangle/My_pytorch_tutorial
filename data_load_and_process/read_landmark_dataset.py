from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode



#1.如何读取csv文件
landmarks_frame = pd.read_csv('faces/face_landmarks.csv')
n = 0 #读取第n+1个记录
# 总之就是  aaa.ix[n]就是aaa对应csv的第(n+1)条记录, aaa.ix[n, a]就是对应csv第(n+1)条记录的第(a+1)个选项
img_name = landmarks_frame.ix[n, 0] #读取相应csv文件的第(n + 1)条记录的第(0 + 1)个元素
landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float') #目前还是一维的numpy数组(???)
landmarks = landmarks.reshape(-1, 2)

'''
print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))
'''




#2. 如何利用csv文件读取后的信息,这里是根据名字展示图片
#image是图片,landmarks这里是[68,2]的数组
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(3)  # pause a bit so that plots are updated
'''

plt.figure()
show_landmarks(io.imread(img_name), landmarks)
plt.show()
'''




#3. 关于torch.utils.data.Dataset
# 就是把csv的读取什么的写成一个类罢了(这里针对了我们这个教程csv文件的内容格式), 它继承是torch.utils.data.Dataset(抽象类),然后实现这个抽象类的方法
# __len__ 是得到dataset的size
# __getitem__支持下标,所以是得到某个下标的对应的具体项
#在__init__部分读取csv,把读取图片的任务给__getitem___,这是memory efficient的因为images不保存在memory at once but read as required
#

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    #看下面的应用
    def __len__(self):
        return len(self.landmarks_frame)

    #支持下标??看下面的的应用
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')
        landmarks = landmarks.reshape(-1, 2)
        #sample就是python的所谓字典
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
'''
#下面csv_file的值是创建的csv文件的地址,那个root_dir在我这个程序用不找,因为csv保存的时候,已经把路径都保存了
face_dataset = FaceLandmarksDataset(csv_file = 'faces/face_landmarks.csv',root_dir= '')

fig = plt.figure()

for i in range(len(face_dataset)): #和__len__函数有关
    sample = face_dataset[i] #貌似和那个类中那个_getitem__函数有关,它支持下标返回??!!

    print(i, sample['image'].shape, sample['landmarks'].shape)

    #下面的代码是plt框架的,这里不细究,大概是每3张图片就一起输出之类的
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 2:
        plt.show()
        break

'''




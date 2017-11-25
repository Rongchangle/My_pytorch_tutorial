from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import read_landmark_dataset
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

'''
紧紧接着read_landmark_dataset1.py这个文件,还是利用了之前文件实现的类,不过这里对这个类的transform参数进行了说明
多了transform这个功能,可以对数据集的图片进行处理,
可能是因为有的数据集图片大小是不一致的,但是某些神经网络偏偏对输入图片有大小固定的要求
'''
face_read_from_1 = read_landmark_dataset.FaceLandmarksDataset(csv_file = 'faces/face_landmarks.csv',root_dir= '')


#1. Rescale, RandomCrop, ToTensor这几个功能小类的实现
class Rescale(object):
    """Rescale the image in a sample to a given size.(作用)

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.(具体如何变)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    #就目前而言,我觉得这个__call__有特殊实现,因为有了这个函数,scale = Rescale(224)宣布类变量,可以直接scale(sample),对sample进行这个函数的操作
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w)) #skimage包的内置函数...

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


'''
实现核心代码就是:
image = image[top: top + new_h, left: left + new_w]
landmarks = landmarks - [left, top]
top和left是根据实际情况确定范围的随机数
'''
class RandomCrop(object):
    """Crop randomly the image in a sample.(作用)

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made. (参数作用)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}







'''
2.上面宣布的类都只有一个功能,要是想要组合起来,可以利用torchvision.transforms.Compose
下面是例子
'''

#定义3个操作,前面两个是简单一个类的操作,第三个操作是组合两个类的操作后得到的
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),RandomCrop(224)])

'''
# 下面是测试上面3个操作的代码
fig = plt.figure()
sample = face_read_from_1[1] #采用之前文件的方式读取(就是transform为none的)
for i, tsfrm in enumerate([scale, crop, composed]): #就是轮流测试3个操作
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    read_landmark_dataset.show_landmarks(**transformed_sample)

plt.show()
'''






#3.下面的读取方式和上个文件不同地方在于那个transform参数
transformed_dataset = read_landmark_dataset.FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',root_dir='',
                                               transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))

#下面是测试
'''
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())
'''



#4. 把数据集以batch(批)的形式取出来,下面是2条记录一个batch(我存放了3张图片,所以2个batch,一个batch有2张,一个batch有1张)
#torch.utils.data.DataLoader 提供batching the data, shuffling the data, Load the data in parallel using multiprocessing workers.
dataloader = DataLoader(transformed_dataset, batch_size=2, shuffle=True, num_workers=4)

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch) #len是batch的大小,在DataLoader那里已经设置
    im_size = images_batch.size(2) #size是batch的形状,比如[2,3,224,224]....size(2) = 224

    #下面的代码我也不看了,居然能将一个batch的图片同时展示
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')
'''
下面的batch内部的图片已经是打乱的了
以下,一个for遍历一次数据集(以batch为单位)
如果改写成多次执行下面的for,那么就遍历多次数据集,但是每次batch分组都很可能不一样
比如第一次{1,3},{2} 第二次可能就是{2,3},{1}
'''
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 0:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break







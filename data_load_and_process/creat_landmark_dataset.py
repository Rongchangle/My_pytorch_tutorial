"""Create a sample face landmarks dataset.

Adapted from dlib/python_examples/face_landmark_detection.py
See this file for more explanation.

Download a trained facial shape predictor from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
import dlib
import glob
import csv
from skimage import io

detector = dlib.get_frontal_face_detector() #dlib人脸检测器
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat') #dlib人脸特征点检测,这个.dat文件要自己下载并且放到对应位置,下载地址在本代码上面部分
num_landmarks = 68

with open('faces/face_landmarks.csv', 'w', newline='') as csvfile: #括号里面那个是制作的csv文件名字
    csv_writer = csv.writer(csvfile)

    header = ['image_name']  #header是python的一个list,这里是csv文件第一行(注解)
    for i in range(num_landmarks):
        header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]

    csv_writer.writerow(header) #以行的方式把一个python list写入csv文件

    for f in glob.glob('faces/*.jpg'): #不用太在乎这句话, 它作用应该是把后缀为jpg的文件遍历一遍而已,换其他方法完全没问题
        img = io.imread(f)  # f是图片文件名字
        dets = detector(img, 1)  # face detection

        # 没有检测到人脸或者检测到多个人脸的图片忽略
        if len(dets) == 1:
            row = [f]

            d = dets[0]
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            for i in range(num_landmarks):
                part_i_x = shape.part(i).x
                part_i_y = shape.part(i).y
                row += [part_i_x, part_i_y]

            csv_writer.writerow(row) #以行的方式写入

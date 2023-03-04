import os
import cv2 as cv
import numpy as np


# def main():
#     # 获取文件路径
#     img_path = [os.path.join('./data', f) for f in os.listdir('./data')]
#     # 读取文件标签
#     faces = [cv.imread(img_path, 0) for img_path in img_path]
#     img = [int(f.split('.')[0]) for f in os.listdir('./data')]
#     # 训练数据,保存数据
#     recognizer = cv.face.LBPHFaceRecognizer_create()
#     recognizer.train(faces, np.array(img))
#     recognizer.write('train.yml')


def main():
    path = [os.path.join('./data', f) for f in os.listdir('./data')]
    face = [cv.imread(path, 0) for path in path]
    img = [int(f.split('.')[0]) for f in os.listdir('./data')]
    recognizer = cv.face.LBPHFaceRecognizer_create()  # opencv-contrib-python
    recognizer.train(face, np.array(img))  # 一组人脸信息,一组人脸对应标签
    recognizer.write('train.yml')


main()

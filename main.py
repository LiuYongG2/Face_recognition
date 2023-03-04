# import os
# import cv2 as cv
# import numpy as np
#
# faces_list = []
# labels = []
# label = 1
# # 遍历data下面的图片,检测并提取出人脸信息
# for f in os.listdir('./data'):
#     # 读取图片
#     img = cv.imread(os.path.join('./data', f), 0)
#     # 提取人脸信息
#     face_classifier = cv.CascadeClassifier(
#         "F:/altext/Python project/text2302/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
#     faces = face_classifier.detectMultiScale(img)
#     if len(faces) == 0:
#         continue
#     x, y, w, h = faces[0]
#     faces_list.append(img[y:y + h, x:x + w])
#     labels.append(label)
#     label += 1
#
# # 训练数据并保存
# recognizer = cv.face.LBPHFaceRecognizer_create()  # opencv-contrib-python
# recognizer.train(faces_list, np.array(labels))  # 一组人脸信息,一组人脸对应标签
# recognizer.write('train.yml')


# import cv2 as cv
#
#
# # 读取图片，并转成灰度图像
# img = cv.imread('./data/img_2.png')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
# # 提取图片中的人脸特征信息
# face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
# faces = face_classifier.detectMultiScale(gray)
#
# # 加载识别器和训练数据
# recognizer = cv.face.LBPHFaceRecognizer_create()
# recognizer.read('train.yml')
#
# # 遍历图片中的人脸
# for x, y, w, h in faces:
#
#     # 识别图片中的人脸，返回标签和置信度
#     img_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#     if confidence > 100:
#         name = 'unknown'
#     else:
#         name = 'JAMES' if img_id == 1 else 'lena'
#
#     # 标出识别出的人名，用圆圈出人脸
#     cv.putText(
#         img=img, org=(x, y), text=name,
#         fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
#         color=(0, 255, 0), thickness=1
#     )
#     cv.circle(
#         img=img, center=(x + w // 2, y + h // 2), radius=w//2,
#         color=(255, 0, 0), thickness=1
#     )
#
# # 展示标记后的图片
# cv.imshow('face', img)
# while True:
#     if cv.waitKey(1) == ord('q'):
#         break
# # 释放内存
# cv.destroyAllWindows()

# import os
# import cv2 as cv
# import numpy as np
#
# # 读取图片，并转成灰度图像
# img = cv.imread('./data/img_2.png')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("打开摄像头失败")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("读帧失败")
#         break
#
#     # 提取图片中的人脸特征信息
#     face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
#     faces = face_classifier.detectMultiScale(gray)
#
#     # 加载识别器和训练数据
#     recognizer = cv.face.LBPHFaceRecognizer_create()
#     recognizer.read('train.yml')
#
#     # 遍历图片中的人脸
#     for x, y, w, h in faces:
#
#         # 识别图片中的人脸，返回标签和置信度
#         img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#         if confidence > 100:
#             name = 'unknown'
#         else:
#             name = 'JAMES' if img_id == 1 else 'lena'
#
#         # 标出识别出的人名，用圆圈出人脸
#         cv.putText(
#             img=img, org=(x, y), text=name,
#             fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
#             color=(0, 255, 0), thickness=1
#         )
#         cv.circle(
#             img=img, center=(x + w // 2, y + h // 2), radius=w // 2,
#             color=(255, 0, 0), thickness=1
#         )
#
# # 展示标记后的图片
# cv.imshow('face', frame)
# while True:
#     if cv.waitKey(1) == ord('q'):
#         break
# # 释放内存
# cv.destroyAllWindows()

'''
# import os
# import cv2 as cv
# import numpy as np
#
# def img_entract_faces(img):
#     gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#     face_classifier = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
#     return face_classifier.detectMultiScale(gray), gray
#
# def main():
#     # 创建识别器，加载训练数据
#     recognizer = cv.face.LBPHFaceRecognizer_create()
#     recognizer.read('train.yml')
#     # 打开摄像头
#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         print('链接摄像头失败')
#     # 取标签与人名关系
#     name_map = {int(f.split('.')[0]): f.split('.')[1] for f in os.listdir('./data')}
#     # 循环每一帧画面
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("读帧失败")
#             break
#     # 人脸检测，取人脸部分
#         faces, gray = img_entract_faces(frame)
#     # 遍历人脸，进行识别
#         for x, y, w, h in faces:
#             img_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#             if confidence > 85:
#                 name = 'unknown'
#             else:
#                 name = name_map[img_id]
#             cv.putText(img=frame, org=(x, y), text=name, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 255, 0), thickness=1)
#             cv.circle(img=frame, center=(x + w//2, y + h//2),radius=w//2, color=(255, 0, 0), thickness=1)
#     # 写出人名与人脸框并展示画面
#         cv.imshow('face', frame)
#
#
#         if cv.waitKey(1) == ord('q'):
#             break
#
#     # 关闭摄像头与窗口
#     cap.release()
#     cv.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()
'''

# import os
# import cv2 as cv
# import numpy as np
#
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("摄像头连接失败")
# # 加载识别器和训练数据
# recognizer = cv.face.LBPHFaceRecognizer_create()
# recognizer.read('train.yml')
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("读帧失败")
#         break
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # 提取图片中的人脸特征信息
#     face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
#     faces = face_classifier.detectMultiScale(gray)
#
#     # 遍历图片中的人脸
#     for x, y, w, h in faces:
#
#         # 识别图片中的人脸，返回标签和置信度
#         img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#         if confidence > 85:
#             name = 'unknown'
#         else:
#             name = 'james' if img_id == 1 else 'lena'
#
#         # 标出识别出的人名，用圆圈出人脸
#         cv.putText(
#             img=frame, org=(x, y), text=name,
#             fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
#             color=(0, 255, 0), thickness=1
#         )
#         cv.circle(
#             img=frame, center=(x + w // 2, y + h // 2), radius=w // 2,
#             color=(255, 0, 0), thickness=1
#         )
#     # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 转为灰度图像，减少计算量
#     face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_alt2.xml")  # 加载级联检测器，人脸特征分类器
#     faces = face_classifier.detectMultiScale(gray)
#     # scaleFactor 搜索窗口的比例系数，越小检测时间越长，默认1.1 1.3
#     # minNeighbors 构造目标的相邻矩形最小的个数，默认 3 8
#     # minSize maxSize 限制目标范围
#     for x, y, w, h in faces:
#         cv.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
#     cv.imshow('frame', frame)
#     if cv.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv.destroyAllWindows()

import os
import cv2 as cv


def main():
    recognizer = cv.face.LBPHFaceRecognizer_create()  # opencv-contrib-python
    recognizer.read('train.yml')
    name_map = {int(f.split('.')[0]): f.split('.')[1] for f in os.listdir('./data')}
    print(name_map)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头连接失败")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读帧失败")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 转为灰度图像，减少计算量
        face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_alt2.xml")  # 加载级联检测器，人脸特征分类器
        faces = face_classifier.detectMultiScale(gray)
        # scaleFactor 搜索窗口的比例系数，越小检测时间越长，默认1.1 1.3
        # minNeighbors 构造目标的相邻矩形最小的个数，默认 3 8
        # minSize maxSize 限制目标范围
        # 遍历图片中的人脸
        for x, y, w, h in faces:

            # 识别图片中的人脸，返回标签和置信度
            img_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence > 85:
                name = 'unknown'
            else:
                name = name_map.get(img_id, 'unknown')

            # 标出识别出的人名，用圆圈出人脸
            cv.putText(
                img=frame, org=(x, y), text=name,
                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                color=(0, 255, 0), thickness=1
            )
            cv.circle(
                img=frame, center=(x + w // 2, y + h // 2), radius=w // 2,
                color=(255, 0, 0), thickness=1
            )

        # 展示标记后的图片
        cv.imshow('face', frame)
        if cv.waitKey(1) == ord('q'):
            break
        # 释放内存
    cap.release()
    cv.destroyAllWindows()


main()

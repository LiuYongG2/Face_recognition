# # -*- coding:utf-8 -*-
# import cv2 as cv
# import os
# import numpy as np
#
#
# # 打开摄像头
# def main():
#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         print("打开摄像头失败")
#     # 输入名字
#     name = input("请输入名字：")
#     print("姓名输入完成，按‘s’键保存人脸信息，按‘q’键退出")
#     # 循环每一帧画面1
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("读帧失败")
#             break
#         # 人脸检测，取人脸部分
#         # 框出人脸
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
#         faces = face_classifier.detectMultiScale(gray)
#         for x, y, w, h in faces:
#             cv.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=1)
#         cv.imshow('face', frame)
#         # 保存并退出
#         a = cv.waitKey(1)
#         if a == ord('s'):
#             x, y, w, h = faces[0]
#             cv.imwrite('./data./2.JJ2.JJ.png', frame[y:y + h, x:x + w])
#         elif a == ord('q'):
#             break
#
#     cap.release()
#     cv.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()

# -*- coding:utf-8 -*-
import cv2 as cv
import os


def img_entract_faces(img):
    gray = cv.cvtColor(img, cv.COLORMAP_CIVIDIS)
    face_classifier = cv.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
    return face_classifier.detectMultiScale(gray), gray


def get_image_name(name):
    name_map = {f.split('.')[1]: int(f.split('.')[0]) for f in os.listdir('./data')}
    if not name_map:
        name_number = 1
    elif name in name_map:
        name_number = name_map[name]
    else:
        name_number = max(name_map.values()) + 1
    return './data/' + str(name_number) + '.' + name + ".jpg"


def save_face(faces, img, name):
    if len(faces) == 0:
        print("没有检测到人脸，请调整")
        return
    if len(faces) > 1:
        print("检测到多个人脸，请调整")
        return
    x, y, w, h = faces[0]
    cv.imwrite(get_image_name(name), img[y:y + h, x:x + w])
    print("录入成功")


# 打开摄像头


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("打开摄像头失败")
    # 输入名字
    name = input("请输入名字：")
    print("姓名输入完成，按‘s’键保存人脸信息，按‘q’键退出")
    # 循环每一帧画面1
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读帧失败")
            break
        # 人脸检测，取人脸部分
        faces, gray = img_entract_faces(frame)
        # 框出人脸
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
        cv.imshow('face', frame)
        # 保存并退出
        a = cv.waitKey(1)
        if a == ord('s'):
            save_face(faces, gray, name)
        elif a == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

# encoding: utf-8
"""
@author: shuxiangguo
@file: tutorial01.py
@time: 2018-12-16 23:16:28
"""

import cv2 as cv
import numpy as np

def get_image_info(image):
	print(type(image))
	print(image.shape)
	print(image.size)
	print(image.dtype)
	pixel_data = np.array(image)
	print(pixel_data)


def video_demo():

	#打开电脑相机
	capture = cv.VideoCapture(0)
	while True:
		ret, frame = capture.read()
		#反转摄像头
		frame = cv.flip(frame, 1)
		cv.imshow("video", frame)
		c = cv.waitKey(50)
		if c == 27:
			break

src = cv.imread("../pictures/reba.jpg")

#创建GUI显示图片
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input iamge", src)
get_image_info(src)
#转成灰度图像
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imwrite('reba2.jpg', gray)

#如果不设置等待多久，会等待下一个用户操作才会关掉
cv.waitKey(0)
cv.destroyAllWindows()
print("Hi python")
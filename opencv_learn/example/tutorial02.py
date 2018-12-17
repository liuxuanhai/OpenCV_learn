# encoding: utf-8
"""
@author: shuxiangguo
@file: tutorial02.py
@time: 2018-12-17 02:36:34
"""

import cv2 as cv
import numpy as np


def access_pixels(image):
	print(image.shape)
	height = image.shape[0]
	width = image.shape[1]
	channels = image.shape[2]
	print("width: %s, hieght: %s, channel: %s" %(width, height, channels))

	for row in range(height):
		for col in range(width):
			for c in range(channels):
					pv = image[row, col, c]
					image[row, col, c] = 255 - pv
	cv.imshow("pixels_demo", image)


def create_image():
	# 3个通道的顺序：blue、green、red
	# img = np.zeros([400, 400, 3], np.uint8)
	# img[:, :, 0] = np.ones([400, 400])*255
	# cv.imshow("new image", img)

	# 生成单通道图像
	img = np.ones([400, 400, 1], np.uint8)
	img = img * 127
	cv.imshow("new image", img)


def inverse(image):
	dst = cv.bitwise_not(image)
	cv.imshow("inverse demo", dst)

src = cv.imread("../pictures/reba.jpg")


# 创建GUI显示图片
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input iamge", src)

# 计时
t1 = cv.getTickCount()
# access_pixels(src)
# create_image()
inverse(src)
t2 = cv.getTickCount()
time = (t2 - t1) / cv.getTickFrequency()
print("Time: %s ms" % time)


# 如果不设置等待多久，会等待下一个用户操作才会关掉
cv.waitKey(0)
cv.destroyAllWindows()

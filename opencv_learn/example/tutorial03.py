# encoding: utf-8
"""
@author: shuxiangguo
@file: tutorial03.py
@time: 2018-12-17 06:33:11
"""

import cv2 as cv
import numpy as np
# import pandas as pd


def color_apace_demo(image):
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	cv.imshow('gray', gray)
	hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	cv.imshow('hsv', hsv)
	yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
	cv.imshow('yuv', yuv)
	Ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
	cv.imshow("Ycrcb", Ycrcb)


def extract_object_demo():
	capture = cv.VideoCapture("red.mp4")
	while True:
		ret, frame = capture.read()
		if ret == False:
			break
		hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
		lower_hsv = np.array([156, 43, 46])
		upper_hsv = np.array([180, 255, 255])
		mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
		cv.imshow("video", frame)
		cv.imshow("mask", mask)
		c = cv.waitKey(40)
		if c == 27:
			break


src = cv.imread("../pictures/reba.jpg")

# 创建GUI显示图片
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input iamge", src)
# color_apace_demo(src)
# extract_object_demo()
b, g, r = cv.split(src)
cv.imshow('blue', b)
cv.imshow('green', g)
cv.imshow('red', r)

src[:, :, 2] = 0
cv.imshow("changed image", src)

src = cv.merge([b, g, r])
cv.imshow("mergerd image", src)

# 如果不设置等待多久，会等待下一个用户操作才会关掉
cv.waitKey(0)
cv.destroyAllWindows()
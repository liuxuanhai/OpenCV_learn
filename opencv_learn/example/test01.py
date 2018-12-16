# encoding: utf-8
"""
@author: shuxiangguo
@file: test01.py
@time: 2018-12-16 23:05:17
"""

import cv2 as cv

src = cv.imread("../pictures/reba.jpg")

# 创建GUI显示图片
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input iamge", src)

# 如果不设置等待多久，会等待下一个用户操作才会关掉
cv.waitKey(0)
cv.destroyAllWindows()
print("Hi python")
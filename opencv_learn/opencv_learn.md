### OpenCV学习笔记
#### 一、环境搭建
**windows + python3.7**
> pip install opencv-python  
> pip install -i https://pypi.douban.com/simple opencv-contrib-python    
>pip install pytesseract

#### 二、课程内容

#### 1.第一节课

知识点小结：

* 读取图片：src = cv.imread("pic/path/xxx.jpg")
* 创建GUI显示图片：cv.namedWindow("xxx", cv.WINDOW_AUTOSIZE)
* 设置等待，否则会已知等待到下一个用户操作：cv.waitKey(0)

代码如下：
```python
import cv2 as cv

src = cv.imread("../pictures/reba.jpg")

#创建GUI显示图片
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input iamge", src)

#如果不设置等待多久，会等待下一个用户操作才会关掉
cv.waitKey(0)
cv.destroyAllWindows()
```

#### 2.第二节课

 知识点小结：

* 打开本机摄像头：cv.VideoCapture(0)
* 翻转摄像头：cv.flip(frame, 1)
* 转成灰度图像：gray = cv.cvtColor(src, cv.COLOR_BGR_AUTOSIZE)

代码如下：
```python
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
```
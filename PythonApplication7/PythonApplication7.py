import cv2
import numpy as np
img = cv2.imread("Test.jpg");
#转为灰度图
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


#直方图均衡化
def func1(img):
	hdj=dict()
	for i in range(256):
		hdj[i]=0

	sp = img.shape
	for i in range(sp[0]):
		for j in range(sp[1]):
			cur = img[i][j]
			hdj[cur] = hdj[cur]+1

	sumP=list()
	#累加概率
	sum = 0
	for i in range(256):
		sum += (hdj[i] / (sp[0]*sp[1]))
		sumP.append(sum)

	#新图
	out = np.copy(img)

	for i in range(sp[0]):
		for j in range(sp[1]):
			out[i][j] = 255* sumP[out[i][j]]
	return out


#高斯核生成函数
def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


#高斯过滤
def func2(img):
	out = np.copy(img)

	#生成高斯核
	kernel = gauss(3,0.5)

	sum = np.sum(kernel)

	sp = img.shape
	for i in range( sp[0] ):
		for j in range(sp[1]):
			if (i-1>=0 and j-1>=0 and i+1<=sp[0]-1 and j+1 <=sp[1]-1):
				a = img[i-1][j-1] * kernel[0][0]
				a += img[i-1][j] * kernel[0][1]
				a += img[i-1][j+1] * kernel[0][2]

				a += img[i][j-1] * kernel[1][0]
				a += img[i][j] * kernel[1][1]
				a += img[i][j+1] * kernel[1][2]

				a += img[i+1][j-1] * kernel[2][0]
				a += img[i+1][j] * kernel[2][1]
				a += img[i+1][j+1] * kernel[2][2]
				out[i][j] = a/sum
	return out

#均值滤波
def func3(img):
	out = np.copy(img)
	kernel = np.asarray([[1,2,1],[2,4,2],[1,2,1]])
	sum = np.sum(kernel)
	sp = img.shape
	for i in range( sp[0] ):
		for j in range(sp[1]):
			if (i-1>=0 and j-1>=0 and i+1<=sp[0]-1 and j+1 <=sp[1]-1):
				a = img[i-1][j-1] * kernel[0][0]
				a += img[i-1][j] * kernel[0][1]
				a += img[i-1][j+1] * kernel[0][2]

				a += img[i][j-1] * kernel[1][0]
				a += img[i][j] * kernel[1][1]
				a += img[i][j+1] * kernel[1][2]

				a += img[i+1][j-1] * kernel[2][0]
				a += img[i+1][j] * kernel[2][1]
				a += img[i+1][j+1] * kernel[2][2]
				out[i][j] = a/sum
	return out


#边缘检测-Robert算子
def func4(img):
	out = np.copy(img)
	sp = img.shape
	img = img.astype("float")
	for i in range( sp[0] ):
		for j in range(sp[1]):
			hdj = img[i][j]
			if(i+1<=sp[0]-1 and j+1<=sp[1]-1):
				out[i][j] = abs(  img[i][j]-img[i+1][j+1] ) + abs( img[i][j+1] - img[i+1][j] )
	return out

#原图
cv2.imshow("Test",img)
#直方图均衡
cv2.imshow("zhi fang tu",func1(img))
#高斯滤波
cv2.imshow("gao si ",func2(img))
#均值滤波
cv2.imshow("junzhi",func3(img))
#Robert边缘检测
cv2.imshow("Robert",func4(img))

cv2.waitKey(0)
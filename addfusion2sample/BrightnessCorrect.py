import cv2
import numpy as np
import math

# tiff图片转换
# img = cv2.imread('input.tif', cv2.IMREAD_UNCHANGED)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# cv2.imwrite('output.jpg', img)



###  线性调整（亮的部分值大，暗的部分值小）
###  以正中间的一行像素值为基础，计算其均值avgref
###  然后其下面的每一行都计算均值，然后计算每一行要乘的权重w = avgref/avg
###  之后每一行乘对应的权重
def brightnessLinearCorrect(imagePath):
    #img = cv2.imread('./simdata/202401140003.jpg')
    img = cv2.imread(imagePath)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img1.shape
    #正中间的一行作为基准
    refRow = int(rows//2)
    avg = sum(img1[refRow,:])/4096
    #之后每一行要成的权重
    w = []
    for row in range(refRow+1,rows):
        w.append(avg/(sum(img1[row,:])/4096))

    for row in range(refRow+1,rows):
        for col in range(0,cols):
            img1[row,col] = w[row-refRow-1]*img1[row,col]

    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
    cv2.imwrite('./simdata/202401140004C.jpg',img1)

###高斯
## 先进行归一化，否则算出都是零
def brightnessGaussCorrect(imagePath):
    img = cv2.imread(imagePath)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img1.shape
    # 最中间的一行均值  作为μ，即亮度最大值
    refRow = int(rows//2)
    avg = sum(img1[refRow, :]) / 4096
    # 取sigma值为1，先试试
    sigma = 2
    gaussRef = 1/(sigma*math.sqrt(2*math.pi))*math.exp(-1*1/(2*sigma*sigma))
    #print(gaussRef)  

    w = []
    for row in range(refRow+1,rows):
        tmpavg = sum(img1[row,:])/4096
        normW = avg/tmpavg
        gaussW = 1/(sigma*math.sqrt(2*math.pi))*math.exp(-1*(normW)**2/(2*sigma*sigma))
        w.append(gaussRef/gaussW)

    for row in range(refRow+1,rows):
        for col in range(0,cols):
            img1[row,col] = w[row-refRow-1]*img1[row,col]

    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('./simdata/202401140003G.jpg', img1)

## 直方图均衡   不行
def euqaProcess(imagePath):
    img = cv2.imread(imagePath)
    res = np.zeros_like(img)
    for i in range(3):
        res[:,:,i] = cv2.equalizeHist(img[:,:,i])
    cv2.imwrite('./simdata/equa.jpg', res)

## clahe 对比度自适应直方图均衡化
def claheProcess(imagePath):
    img = cv2.imread(imagePath)
    res = np.zeros_like(img)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    for i in range(3):
        res[:,:,i] = clahe.apply(img[:,:,i])
    cv2.imwrite('./simdata/clahe.jpg', res)

###
def illum(path):
    img = cv2.imread(path)
    heigth,wigth = img.shape[0:2]
    mask = np.zeros(img.shape, dtype = np.uint8)
    #mask[0:heigth//2,:,:] = 255
    res = cv2.illuminationChange(img,mask=mask,alpha=3.2,beta=0.02)
    #img[heigth//2:heigth,:] = res[heigth//2:heigth,:]
    cv2.imwrite('test.jpg', res)




if __name__ == '__main__':
    path = "./simdata/202401140003.jpg"
    #brightnessLinearCorrect(path)
    brightnessGaussCorrect(path)
    #euqaProcess(path)
    #claheProcess(path)
    #illum(path)
import numpy as np
import cv2
import argparse
import os
import copy
import math
def getMask(image):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imageGray,0,255,cv2.THRESH_BINARY)
    return mask
def simpleAdd(mask,result,imageB):
    pass

####类似于改变亮度
def getLinearArea(mask,imageB,result):
    edges = cv2.Canny(mask, 100, 200)
    bimage, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    interArea = np.zeros_like(imageB)
    for r in range(y,y+h):
        for c in range(x,x+w):
            if result[r,c,0]==0 and result[r,c,1]==0 and result[r,c,1]==0:
                interArea[r,c,:] = imageB[r,c,:]
            elif imageB[r,c,0]==0 and imageB[r,c,1]==0 and imageB[r,c,2]==0:
                interArea[r,c,:] = result[r,c,:]
            else:
                interArea[r,c,:] = result[r,c,:] * ((r-y)/h) + imageB[r,c,:] * (1-((r-y)/h))
                #/ 2 +  (  result[r,c,:] * ((c-x)/w) + imageB[r,c,:] * (1-((c-x)/w)) ) / 2
    return x,y,w,h,interArea

#########
###羽化融合
def getGaussianBlur(mask):
    gaussMask = cv2.GaussianBlur(mask,(211,211),0)
    gaussMask = gaussMask.astype(np.float32) / 255
    return gaussMask
###
def getGaussianRes(gaussMask,imageB,result,mask):
    ##重叠区域
    interArea = np.zeros_like(imageB)
    ##变换dtype
    imageB = imageB.astype(np.float32) / 255
    result = result.astype(np.float32) / 255
    interArea = interArea.astype(np.float32) / 255
    ## 找到重叠区域
    edges = cv2.Canny(mask, 100, 200)
    bimage, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    for r in range(y, y + h):
        for c in range(x, x + w):
            alpha = gaussMask[r,c]
            #print(alpha)
            if result[r, c, 0] == 0 and result[r, c, 1] == 0 and result[r, c, 1] == 0:
                interArea[r, c, :] = imageB[r, c, :]
            elif imageB[r, c, 0] == 0 and imageB[r, c, 1] == 0 and imageB[r, c, 2] == 0:
                interArea[r, c, :] = result[r, c, :]
            else:
                interArea[r, c, :] = result[r, c, :] * (alpha) + imageB[r, c, :] * (1-alpha)
                #32f类型的数据进行渐变蒙版
                #interArea[r, c, :] = result[r, c, :] * ((r - y) / h) + imageB[r, c, :] * (1 - ((r - y) / h))
    res = (interArea * 255).astype(np.uint8)
    return x, y, w, h, res

########亮度


######################################
########图片处理
def isProcess(result,imageB,moveX,moveY):
    #### 把边缘2像素 裁去
    imageBNew = imageB[2:imageB.shape[0]-2,2:imageB.shape[1]-2,:]


    ##### 粘贴，为了求并集部分的mask
    tempResult = result.copy()
    if moveX < 0 and moveY < 0:
        tempResult[abs(moveY)+2:abs(moveY)+imageB.shape[0]-2, abs(moveX)+2:abs(moveX)+imageB.shape[1]-2] = imageBNew
    elif moveX < 0 and moveY > 0:
        tempResult[2:imageB.shape[0]-2, abs(moveX)+2:abs(moveX)+imageB.shape[1]-2] = imageBNew
    elif moveX > 0 and moveY < 0:
        tempResult[abs(moveY)+2:abs(moveY)+imageB.shape[0],2:imageB.shape[1]-2] = imageBNew
    else:
        tempResult[2:imageB.shape[0]-2, 2:imageB.shape[1]-2] = imageBNew

    #未作处理的结果如下
    #cv2.imwrite("temp34.png",tempResult)
    #####  imageB位置对齐  获取b图的mask
    paddingX = abs(moveX)
    paddingY = abs(moveY)
    if moveX < 0 and moveY < 0:
        imageBPad = cv2.copyMakeBorder(imageBNew, paddingY+2, result.shape[0]-imageB.shape[0]-paddingY+2, paddingX+2,
                                    result.shape[1]-imageB.shape[1]-paddingX+2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif moveX < 0 and moveY > 0:
        imageBPad = cv2.copyMakeBorder(imageBNew, 2, result.shape[0] - imageB.shape[0]+2, paddingX+2,
                                    result.shape[1] - imageB.shape[1] - paddingX+2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif moveX > 0 and moveY < 0:
        imageBPad = cv2.copyMakeBorder(imageBNew, paddingY+2, result.shape[0] - imageB.shape[0] - paddingY+2, 2,
                                    result.shape[1] - imageB.shape[1]+2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        imageBPad = cv2.copyMakeBorder(imageBNew, 2, result.shape[0] - imageB.shape[0]+2,2,
                                    result.shape[1] - imageB.shape[1]+2, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    #maskTempResult = getMask(tempResult)
    maskImageB = getMask(imageBPad)
    maskResult = getMask(result)
    ##交集  并集   并-交
    maskInter = np.uint8(maskImageB & maskResult)
    #maskUnion = np.uint8(maskImageB | maskResult)
    #maskDiff = np.uint8(maskUnion - maskInter)

    ##测试高斯
    #gaussMask = getGaussianBlur(maskInter)
    #x,y,w,h,interArea = getGaussianRes(gaussMask,imageBPad,result,maskInter)
    ###########

    ##线性 渐变 蒙版
    x,y,w,h,interArea = getLinearArea(maskInter, imageBPad, result)

    # 把处理后的交集部分的图片贴到 直接拼接好的两张图像上
    tempResult[y+2:y+h-2, x+2:x+w-4] = interArea[y+2:y+h-2, x+2:x+w-4]
    cv2.imwrite("jianbian-32f.png", tempResult)
    pass
    return tempResult




##############################################################################################
#################################图像拼接
def detectAndDescribe(image,kpsAlgorithm=0,siftNfeatures=500):
    if kpsAlgorithm==0:
        descriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=5000)
    elif kpsAlgorithm==1:
        descriptor = cv2.xfeatures2d.SIFT_create(nfeatures=siftNfeatures)
    elif kpsAlgorithm==2:
        descriptor = cv2.ORB_create()
    elif kpsAlgorithm==3:
        descriptor = cv2.BRISK_create()
    # 检测特征点 并计算描述子
    # kps为关键点列表 其中的信息其实很多 比如angle,pt,size等等
    (kps, features) = descriptor.detectAndCompute(image, None)
    # 只需要kps中的pt即可 pt是关键点坐标 将结果转换成NumPy数组
    # kp.pt相当于一个二维坐标
    kps = np.float32([kp.pt for kp in kps])
    # 返回特征点集，及对应的描述特征
    return (kps, features)
# matchKeypoints对两张图像的特征点进行配对
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio = 0.75, reprojThresh = 4.0):
    matcher = cv2.BFMatcher_create(normType=cv2.NORM_L2)
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))
    # 当筛选后的匹配对大于4时，计算视角变换矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # 根据对应点可以计算单应矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        # 返回结果
        return (matches, H, status)
    # 如果匹配对小于4时，返回None
    return None
def correct_H(H,w,h):
    corner_pts = np.array([[[0, 0], [w, 0], [0, h], [w, h]]], dtype=np.float32)
    min_out_w, min_out_h = cv2.perspectiveTransform(corner_pts, H)[0].min(axis=0).astype(np.int_)
    if min_out_w < 0 and min_out_h < 0:
        H[0, :] -= H[2, :] * min_out_w
        H[1, :] -= H[2, :] * min_out_h
    elif min_out_w < 0 and min_out_h > 0:
        H[0, :] -= H[2, :] * min_out_w
    elif min_out_w > 0 and min_out_h < 0:
        H[1, :] -= H[2, :] * min_out_h
    moveX = min_out_w
    moveY = min_out_h
    return H, moveX, moveY
def stitchImage(imageAPath,imageBPath,imageSavePath='.\\result.png',
                kpsAlgorithm=1,siftNfeatures=500,ratio=0.75, reprojThresh=4.0 ,warpedSize=0.7):
    # 读取两张待拼接的两张图片
    imageA = cv2.imread(imageAPath)
    imageB = cv2.imread(imageBPath)
    # 这里的情况只考虑两张拼接图像是相同size的
    h,w,c = imageA.shape
    # 通过detectAndDescribe函数得到关键点（这里的kpsA B是坐标）及其对应的descriptors描述子(featuresA B其实这里不应该写成特征的)
    (kpsA, featuresA) = detectAndDescribe(imageA,kpsAlgorithm,siftNfeatures)
    (kpsB, featuresB) = detectAndDescribe(imageB,kpsAlgorithm,siftNfeatures)
    # 得到两张图片的特征点后，需要对这两张图片的特征点进行配对
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    # 先判断是否是空，及两张图片是否无法拼接
    if M is None:
        return None
    (matches, H, status) = M
    # 计算新的单应矩阵H，矫正变换后图像显示不全的问题
    (H, moveX, moveY) = correct_H(H,w,h)
    # 图像变换，这里选用两个图片长宽相加，是因为考虑到多张图片拼接的情况，过程中有裁剪会导致图像大小变得不一样
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + int(warpedSize*imageB.shape[1]), imageA.shape[0] + int(warpedSize*imageB.shape[0])))
    resultTmp = copy.deepcopy(result)

    ##########################################
    ###########################直接拷贝
    ##########################################
    # if moveX < 0 and moveY < 0:
    #     result[abs(moveY)+2:abs(moveY)+imageB.shape[0]-2, abs(moveX)+2:abs(moveX)+imageB.shape[1]-2] = imageB[2:imageB.shape[0]-2,2:imageB.shape[1]-2]
    # elif moveX < 0 and moveY > 0:
    #     result[2:imageB.shape[0]-2, abs(moveX)+2:abs(moveX)+imageB.shape[1]-2] = imageB[2:imageB.shape[0]-2,2:imageB.shape[1]-2]
    # elif moveX > 0 and moveY < 0:
    #     result[abs(moveY)+2:abs(moveY)+imageB.shape[0]-2, 2:imageB.shape[1]-2] = imageB[2:imageB.shape[0]-2,2:imageB.shape[1]-2]
    # else:
    #     result[2:imageB.shape[0]-2, 2:imageB.shape[1]-2] = imageB[2:imageB.shape[0]-2,2:imageB.shape[1]-2]


    #################################
    ###########处理
    processedRes = isProcess(resultTmp,imageB,moveX,moveY)

    rows, cols = np.where(processedRes[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    processedRes = processedRes[min_row:max_row, min_col:max_col, :]
    print(processedRes.dtype)
    cv2.imwrite(imageSavePath, processedRes)

if __name__ == '__main__':
    # 两张图片拼接
    testImage1Path = "./simdata/202401140003.jpg"
    testImage2Path = "./simdata/202401140004.jpg"
    stitchImage(testImage1Path,testImage2Path,imageSavePath="all1.jpg",kpsAlgorithm=1)

    # 文件夹内的所有图片 按顺序拼接
    # path = "D:\\Code\\PyCharm\\PyProject\\v2Reconstruct\\simdata\\simtest\\"
    # images = os.listdir(path)
    # imglist = []
    # for i in images:
    #     imglist.append(path+i)
    # imgnums = len(imglist)
    # img1 = imglist[0]
    # img2 = imglist[1]
    # stitchImage(img1,img2,imageSavePath="res.jpg",kpsAlgorithm=1)
    # for i in range(2, imgnums):
    #     imgi = imglist[i]
    #     print(i)
    #     stitchImage("res.jpg",imgi,imageSavePath="res.jpg",kpsAlgorithm=1)

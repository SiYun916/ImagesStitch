import numpy as np
import cv2
import argparse
import os
import copy

# 全局变量，图像A只做一次gamma变换，否则会出现warp后的图像一直进行gamma变换
imageAGammaCount=0
# 同上，图像A只做一次自适应的直方图均衡
imageAEqualCount=0
# 同上，图像A只做一次畸变矫正
imageADisCorCount=0
#######################################
###############对图像的简单处理
#######################################
# 彩色图像R、G、B三通道均衡化（全局直方图均衡化）
# 效果不好，暂时都先别用
def RGBEqualize(image):
    (b, g, r) = cv2.split(image)
    equal_b = cv2.equalizeHist(b)
    equal_g = cv2.equalizeHist(g)
    equal_r = cv2.equalizeHist(r)
    result = cv2.merge((equal_b, equal_g, equal_r))
    return result
#  自适应彩色图像直方图均衡
def adaptiveHisEqual(image):
    # 转化到ycrcb空间
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    # 通道分离
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, image)
    return image
# 伽马变换
# 大于1调亮，低于1调暗
# opencv默认imread读取的图像的格式是BGR
# 目前只是针对第二张及其之后的图像进行调整，无法为每个拼接进行调整
def gammaAdjust(image,gamma=1.0):
    imagef = image.astype(np.float32)/255
    result = (np.power(imagef, 1/gamma)*255).astype(np.uint8)
    return result
# 畸变矫正
def distortionCor(image,camMat,distCoeffs):
    h, w = image.shape[:2]
    K = np.zeros((3, 3))
    K[0, 0] = camMat[0]
    K[1, 1] = camMat[1]
    K[0, 2] = camMat[2]
    K[1, 2] = camMat[3]
    K[2, 2] = 1
    distCoeffs = np.float32(distCoeffs)
    retval, validPixROI	= cv2.getOptimalNewCameraMatrix(cameraMatrix=K,distCoeffs=distCoeffs,imageSize=(h,w),alpha=0)
    src = image
    dst = cv2.undistort(src=src, cameraMatrix=K, distCoeffs=distCoeffs,newCameraMatrix=retval)
    return dst
#######################################
###############添加左上角添加文字
#######################################
# 获取自适应的文字大小和文字高度
def getOptimalFontScale(text, width):
    for scale in reversed(range(0, 100, 1)):
        textsize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale/10, thickness=1)
        newWidth = textsize[0][0]
        newHeight = textsize[0][1]
        if (newWidth <= width):
            return (scale/10,newHeight)
    return (1,75)
def addText(stitchedImage,pathList,savepath):
    # 获取图片名称
    imageNameList = []
    for imagepath in pathList:
        (path, imageName) = os.path.split(imagepath)
        imageName = imageName[-8:]
        imageNameList.append(imageName)
    # 生成文字
    text = "Stitched Images: "
    for imagename in imageNameList:
        (name, suffix) = os.path.splitext(imagename)
        text += name + " "
    # 读取图片
    image = cv2.imread(stitchedImage)
    # 获取自适应的字体
    fontscale = 3 * (image.shape[1] // 5)
    (fontSize, borderHeight) = getOptimalFontScale(text, fontscale)
    # 在上方添加黑色像素
    image = cv2.copyMakeBorder(image, borderHeight + 35, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    # 图片上加字
    cv2.putText(image, text, (0, borderHeight + 10), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), 2)
    cv2.imwrite(savepath,image)
#######################################
###############图像融合
#######################################
def imageFusion(result,imageB,moveX,moveY):
    #记录未增边界前imageB的长宽
    bHeight,bWidth = imageB.shape[0:2]
    #新图片
    resultNew = np.zeros([result.shape[0], result.shape[1], 3])
    # 边界部分
    picrow = 0
    piccol = 0
    #给imageB新增边界，保证重叠位置对应
    paddingX = abs(moveX)
    paddingY = abs(moveY)
    if moveX < 0 and moveY < 0:
        imageB = cv2.copyMakeBorder(imageB, paddingY, result.shape[0] - imageB.shape[0] - paddingY, paddingX,
                                    result.shape[1] - imageB.shape[1] - paddingX, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        picrow = np.copy(imageB[abs(moveY):abs(moveY) + bHeight, bWidth - 2:bWidth + 2])
        piccol = np.copy(imageB[bHeight-2:bHeight+2,abs(moveX):abs(moveX)+bWidth])
    elif moveX < 0 and moveY > 0:
        imageB = cv2.copyMakeBorder(imageB, 0, result.shape[0] - imageB.shape[0], paddingX,
                                    result.shape[1] - imageB.shape[1] - paddingX, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        picrow = np.copy(imageB[moveY:moveY + bHeight, bWidth - 2:bWidth + 2])
        piccol = np.copy(imageB[moveY - 2:moveY + 2, abs(moveX):abs(moveX) + bWidth])
    elif moveX > 0 and moveY < 0:
        imageB = cv2.copyMakeBorder(imageB, paddingY, result.shape[0] - imageB.shape[0] - paddingY, 0,
                                    result.shape[1] - imageB.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        picrow = np.copy(imageB[abs(moveY):abs(moveY) + bHeight, moveX-2:moveX-2])
        piccol = np.copy(imageB[bHeight-2:bHeight+2,moveX:moveX+bWidth])
    else:
        imageB = cv2.copyMakeBorder(imageB, 0, result.shape[0] - imageB.shape[0],0,
                                    result.shape[1] - imageB.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        picrow = np.copy(imageB[moveY:moveY+bHeight, moveX-2:moveX-2])
        piccol = np.copy(imageB[moveY-2:moveY+2, moveX:moveX+bWidth])

    #寻找重叠的边界
    rows,cols = imageB.shape[:2]
    left = 0
    right = 0
    for col in range(0,cols):
        if imageB[:,col].any() and result[:,col].any():
            left = col
            break
    for col in range(cols-1,0,-1):
        if imageB[:, col].any() and result[:, col].any():
            right = col
            break

    #
    # print(left,right)
    # crossWeight = abs(left-right)
    #融合
    for row in range(0,rows):
        for col in range(0,cols):
            if not result[row, col].any():
                resultNew[row, col, :] = imageB[row, col, :]
            elif not imageB[row, col].any():
                resultNew[row, col, :] = result[row, col, :]
            else:
                w1 = float(abs(col - left))
                #alpha = w1 / crossWeight
                alpha = 0.5
                resultNew[row, col, :] = result[row, col, :] * (1-alpha) + \
                                      imageB[row, col, :] * alpha

    #边界部分处理
    if moveX < 0 and moveY < 0:
        resultNew[abs(moveY):abs(moveY) + bHeight, bWidth - 2:bWidth + 2] = picrow
        resultNew[bHeight - 2:bHeight + 2, abs(moveX):abs(moveX) + bWidth] = piccol
    elif moveX < 0 and moveY > 0:
        resultNew[moveY:moveY + bHeight, bWidth - 2:bWidth + 2] = picrow
        resultNew[moveY - 2:moveY + 2, abs(moveX):abs(moveX) + bWidth] = piccol
    elif moveX > 0 and moveY < 0:
        resultNew[abs(moveY):abs(moveY) + bHeight, moveX - 2:moveX - 2] = picrow
        resultNew[bHeight - 2:bHeight + 2, moveX:moveX + bWidth] = piccol
    else:
        resultNew[moveY:moveY + bHeight, moveX - 2:moveX - 2] = picrow
        resultNew[moveY - 2:moveY + 2, moveX:moveX + bWidth] = piccol
    return resultNew

#######################################
###############图像拼接流程
#######################################
def detectAndDescribe(image,kpsAlgorithm=0):
    # SIFT特征点提取
    # 参数列表 （不全）
    # nfeatures 特征点数目 算法对检测出的特征点排名 返回最好的nfeatures个特征点 默认为0
    # contrastThreshold 过滤掉较差的特征点的对阈值 contrastThreshold越大 返回的特征点越少 默认0.04
    # edgeThreshold 过滤掉边缘效应的阈值 edgeThreshold越大，特征点越多（被过滤掉的越少）默认 10
    # sigma 金字塔第0层图像高斯滤波系数 默认1.6
    #descriptor = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    if kpsAlgorithm==0:
        descriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=5000)
    elif kpsAlgorithm==1:
        descriptor = cv2.xfeatures2d.SIFT_create(nfeatures=500)
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
    # 建立匹配器
    # 匹配器有两个BFMatcher和FlannBasedMatcher
    ################################
    # FlannBasedMatcher用法和参数
    # cv2.FlannBasedMatcher(index_params, search_params) 两个参数都是字典类型的
    #########################
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # FLANN_INDEX_LINEAR = 0   ：  线性暴力(brute-force)搜索
    # FLANN_INDEX_KDTREE = 1   ：  随机kd树，平行搜索。默认trees=4
    # FLANN_INDEX_KMEANS = 2   ：  层次k均值树。默认branching=32，iterations=11，centers_init = CENTERS_RANDOM, cb_index =0.2
    # FLANN_INDEX_LSH = 6      ：  multi-probe LSH方法
    #########################
    # search_params = dict(checks=32, eps=0, sorted=True)
    # checks为int型，是遍历的次数，一般只改变这个参数
    ################################
    #BFMatcher用法和参数
    #cv::BFMatcher::BFMatcher	(int normType = NORM_L2,bool crossCheck = false )
    #########################
    # 第一个参数是normType，它指定要使用的距离测量。
    # 默认情况下为 cv2.NORM_L2 。对于SIFT, SURF等（也有 cv2.NORM_L1）很有用。
    # 对于基于二进制字符串的描述符，例如ORB，BRIEF，BRISK等，应使用cv2.NORM_HAMMING ，该函数使用汉明距离作为度量。
    # 如果ORB使用WTA_K == 3或 4，则应使用 cv.NORM_HAMMING2。
    #########################
    # 第二个 似乎不太重要

    # 暴力
    matcher = cv2.BFMatcher_create(normType=cv2.NORM_L2)
    # flann
    # indexParams = dict(algorithm=1,trees=5)
    # searchParams = dict(checks=32)
    # matcher = cv2.FlannBasedMatcher(indexParams,searchParams)
    # 创建好匹配器后，使用.match()返回最佳匹配；或者是.knnMatch()返回k个最佳匹配，其中k为指定的
    # 使用KNN检测来自A、B图的特征匹配对，K=2
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    # rawMatches中已经是匹配好的对应点了 下面需要对其进行筛选
    matches = []
    # matches存放筛选的匹配点  matches中是类似于(320,10)的这种数据  320和10都是keypoints的索引
    # 索引10对应的kpsA中的坐标queryIdx，索引320是对应的kpsB中的坐标trainIdx
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

# correct_H函数目的是矫正变换后图像显示不全的问题
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
    # 下面是计算投影图像的宽高
    # correct_w, correct_h = cv2.perspectiveTransform(corner_pts, H)[0].max(axis=0).astype(np.int_)
    # 返回平移的值
    moveX = min_out_w
    moveY = min_out_h
    return H, moveX, moveY
# transformA参数
# imageAPath和imageBPath是图片的路径 传进来再读取
# path 图片的保存路径
# ratio 筛选好的匹配对的参数
# reprojThresh 计算单应矩阵需要的参数

# 先将第一张图片变换到第二张图片的坐标系下，保存仿射变换后的图像
# 第一次提取默认使用SURF算法
def stitchImage(imageAPath,imageBPath,imageSavePath='.\\result.png',
                kpsAlgorithm=1,ratio=0.75, reprojThresh=4.0 ,warpedSize=0.7,
                gammaAlgorithm=0,gamma=1.0,
                adaptiveEqual=0,
                distorCorrect=0,
                camMat=[1.46489094e+03,1.46358621e+03,3.53922761e+02,2.07792630e+02],
                distCoeffs=[-1.11169695e-01 ,1.34520353e+01,1.12320430e-02,7.68615307e-02,0],
                doImageFusion=0):
    # 读取两张待拼接的两张图片
    imageA = cv2.imread(imageAPath)
    imageB = cv2.imread(imageBPath)
    # 是否进行gamma变换
    global imageAGammaCount
    if gammaAlgorithm:
        if imageAGammaCount < 1:
            imageAGammaCount += 1
            imageA = gammaAdjust(imageA,gamma)
        imageB = gammaAdjust(imageB,gamma)
    # 是否进行自适应直方图均衡
    global imageAEqualCount
    if adaptiveEqual:
        if imageAEqualCount < 1:
            imageAEqualCount += 1
            imageA = adaptiveHisEqual(imageA)
        imageB = adaptiveHisEqual(imageB)
    # 畸变矫正
    global  imageADisCorCount
    if distorCorrect:
        if imageADisCorCount < 1:
            imageADisCorCount += 1
            imageA = distortionCor(imageA,camMat,distCoeffs)
        imageB = distortionCor(imageB,camMat, distCoeffs)
    # 这里的情况只考虑两张拼接图像是相同size的
    h,w,c = imageA.shape
    # 通过detectAndDescribe函数得到关键点（这里的kpsA B是坐标）及其对应的descriptors描述子(featuresA B其实这里不应该写成特征的)
    (kpsA, featuresA) = detectAndDescribe(imageA,kpsAlgorithm)
    (kpsB, featuresB) = detectAndDescribe(imageB,kpsAlgorithm)
    # 得到两张图片的特征点后，需要对这两张图片的特征点进行配对
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    # 先判断是否是空，及两张图片是否无法拼接
    if M is None:
        return None
    (matches, H, status) = M
    # 计算新的单应矩阵H，矫正变换后图像显示不全的问题
    # moveX moveY应该是没用了  并不是单纯的平移  之后再改改？
    (H, moveX, moveY) = correct_H(H,w,h)
    # 图像变换，这里选用两个图片长宽相加，是因为考虑到多张图片拼接的情况，过程中有裁剪会导致图像大小变得不一样
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + int(warpedSize*imageB.shape[1]), imageA.shape[0] + int(warpedSize*imageB.shape[0])))
    resultTmp = copy.deepcopy(result)
    # 下面是调整,和去除拼接边界的缝隙。思路是直接拷贝warp后的图的边界3像素区域放到最终图上
    # 如果x,y都是负的 都需要平移 那么imageB也跟着平移
    # 其中一个是正的 另一个为负  只移动一个方向即可
    if moveX < 0 and moveY < 0:
        # 拷贝边界，右下平移后要考虑四个边界
        picrow1 = np.copy(resultTmp[abs(moveY):abs(moveY)+imageB.shape[0], abs(moveX)-2:abs(moveX)+2])
        picrow2 = np.copy(resultTmp[abs(moveY):abs(moveY)+imageB.shape[0], abs(moveX)+imageB.shape[1]-2:abs(moveX)+imageB.shape[1]+2])
        piccol1 = np.copy(resultTmp[abs(moveY)-2:abs(moveY)+2, abs(moveX):abs(moveX)+imageB.shape[1]])
        piccol2 = np.copy(resultTmp[abs(moveY)+imageB.shape[0]-2:abs(moveY)+imageB.shape[0]+2, abs(moveX):abs(moveX)+imageB.shape[1]])
        # 直接拷贝imageB
        if doImageFusion:
            result = imageFusion(resultTmp,imageB,moveX,moveY)
        else:
            result[abs(moveY):abs(moveY)+imageB.shape[0], abs(moveX):abs(moveX)+imageB.shape[1]] = imageB
        # 拷贝边界至结果
        result[abs(moveY):abs(moveY)+imageB.shape[0], abs(moveX)-2:abs(moveX)+2] = picrow1
        result[abs(moveY):abs(moveY)+imageB.shape[0],abs(moveX)+imageB.shape[1]-2:abs(moveX)+imageB.shape[1]+2] = picrow2
        result[abs(moveY)-2:abs(moveY)+2, abs(moveX):abs(moveX)+imageB.shape[1]] = piccol1
        result[abs(moveY)+imageB.shape[0]-2:abs(moveY)+imageB.shape[0]+2,abs(moveX):abs(moveX)+imageB.shape[1]] = piccol2
    elif moveX < 0 and moveY > 0:
        # 向右平移，考虑左右边界和下边界
        picrow1 = np.copy(resultTmp[moveY:moveY+imageB.shape[0],abs(moveX)-2:abs(moveX)+2])
        picrow2 = np.copy(resultTmp[moveY:moveY+imageB.shape[0],imageB.shape[1]+abs(moveX)-2:imageB.shape[1]+abs(moveX)+2])
        piccol1 = np.copy(resultTmp[imageB.shape[0]-2:imageB.shape[0]+2,abs(moveX):abs(moveX)+imageB.shape[1]])

        if doImageFusion:
            result = imageFusion(resultTmp,imageB,moveX,moveY)
        else:
            result[0:0 + imageB.shape[0], abs(moveX):abs(moveX) + imageB.shape[1]] = imageB

        result[moveY:moveY+imageB.shape[0], abs(moveX) - 2:abs(moveX) + 2] = picrow1
        result[moveY:moveY+imageB.shape[0], imageB.shape[1] + abs(moveX) - 2:imageB.shape[1] + abs(moveX) + 2] = picrow2
        result[imageB.shape[0] - 2:imageB.shape[0] + 2, abs(moveX):abs(moveX) + imageB.shape[1]] = piccol1

    elif moveX > 0 and moveY < 0:
        # 直接深拷贝重叠的边界，替换掉缝隙
        # 向下平移考虑上下边界和右边界
        picrow = np.copy(resultTmp[abs(moveY):abs(moveY)+imageB.shape[0],imageB.shape[1]-2:imageB.shape[1]+2])
        piccol1 = np.copy(resultTmp[abs(moveY)-2:abs(moveY)+2,moveX:moveX+imageB.shape[1]])
        piccol2 = np.copy(resultTmp[imageB.shape[0]+abs(moveY)-2:imageB.shape[0]+abs(moveY)+2,moveX:moveX+imageB.shape[1]])

        if doImageFusion:
            result = imageFusion(resultTmp,imageB,moveX,moveY)
        else:
            result[abs(moveY):abs(moveY) + imageB.shape[0], 0:0 + imageB.shape[1]] = imageB

        result[abs(moveY):abs(moveY)+imageB.shape[0], imageB.shape[1]-2:imageB.shape[1]+2] = picrow
        result[abs(moveY)-2:abs(moveY)+2,moveX:moveX+imageB.shape[1]] = piccol1
        result[imageB.shape[0] + abs(moveY) - 2:imageB.shape[0] + abs(moveY) + 2, moveX:moveX+imageB.shape[1]] = piccol2
        # 先深拷贝重叠的边界，重点在找好边界值
        # seemrowMax = max_tmp_row
        # seemcolMin = min_tmp_col
        # for row in range(abs(moveY),max_tmp_row):
        #     if result[row,imageB.shape[1]+5,0]==0 or result[row,imageB.shape[1]-5,0]==0:
        #         seemrowMax=row
        #         break
        # for col in reversed(range(min_tmp_col,imageB.shape[1])):
        #     if result[abs(moveY)-5,col,0]==0 or result[abs(moveY)+5,col,0]==0:
        #         seemcolMin=col
        #         break
        #picrow = np.copy(result[abs(moveY):seemrowMax, imageB.shape[1] - 5:imageB.shape[1] + 5])
        #piccol = np.copy(result[abs(moveY)-5:abs(moveY)+5, seemcolMin:imageB.shape[1]])
        #result[abs(moveY):abs(moveY) + imageB.shape[0], 0:0 + imageB.shape[1]] = imageB
        #result[abs(moveY):seemrowMax, imageB.shape[1] - 5:imageB.shape[1] + 5] = picrow
        #result[abs(moveY)-5:abs(moveY)+5, seemcolMin:imageB.shape[1]] = piccol
    else:
        # 不用平移，考虑右边界和下边界即可
        picrow = np.copy(resultTmp[moveY:moveY+imageB.shape[0],imageB.shape[1]-2:imageB.shape[1]+2])
        piccol = np.copy(resultTmp[imageB.shape[0]-2:imageB.shape[1]+2,moveX:moveX+imageB.shape[1]])
        if doImageFusion:
            result = imageFusion(resultTmp,imageB,moveX,moveY)
        else:
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        result[moveY:moveY+imageB.shape[0], imageB.shape[1] - 2:imageB.shape[1] + 2] = picrow
        result[imageB.shape[0] - 2:imageB.shape[1] + 2, moveX:moveX+imageB.shape[1]] = piccol
    # 裁剪黑边
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    result = result[min_row:max_row, min_col:max_col, :]
    cv2.imwrite(imageSavePath, result)


# 先不考虑拼接的顺序
# 默认用SURF算法
if __name__ == '__main__':
    #文件路径
    # testImage1Path = "./simdata/202401140001.jpg"
    # testImage2Path = "./simdata/202401140003.jpg"
    # testImage1Path = "./data/Vis4096/test2/1.jpg"
    # testImage2Path = "./data/Vis4096/test2/2.jpg"
    # testImage3Path = "./data/Vis4096/test2/3.jpg"
    # testImage4Path = "./data/Vis2048/test3/4.jpg"
    # stitchImage(testImage1Path,testImage2Path,imageSavePath="res7.jpg",kpsAlgorithm=1,doImageFusion=1)
    # stitchImage('.\\transformed.png', testImage3Path, imageSavePath='.\\result.png')
    #stitch(testImage2Path, transformedImagePath='.\\transformed.png', resultImageSavePath=".\\result.png")
    # transformImage(".\\result.png", testImage3Path, interImageSavePath='.\\transformed.png')
    # stitch(testImage3Path, transformedImagePath='.\\transformed.png', resultImageSavePath=".\\result.png")
    # transformImage(".\\result.png", testImage4Path, interImageSavePath='.\\transformed.png')
    # stitch(testImage4Path, transformedImagePath='.\\transformed.png', resultImageSavePath=".\\result.png")



    ############################################
    # 参数设置   不考虑拼接的顺序情况
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", "-l", type=str, nargs='+', help="待拼接的图片路径")
    parser.add_argument("--kpsAlgorithm",type=int,help="特征点提取算法，默认是SIFT",
                        default=1)
    parser.add_argument("--savepath", "-sp", type=str, help="最终结果保存路径",
                        default=".\\result.png")
    parser.add_argument("--warpedSize", type=float, help="如果变换后的图片不完整，调整该参数",
                        default=0.7)
    parser.add_argument("--gammaAlgorithm",type=int,help="是否进行gamma变换（全部），0->NO，1->YES",
                        default=0)
    parser.add_argument("--gamma",type=float,help="gamma变换的参数，大于1变亮",
                        default=1.0)
    parser.add_argument("--adaptiveEqual",type=int,help="是否进行自适应直方图均衡（全部），0->NO，1->YES",
                        default=0)
    parser.add_argument("--distorCorrect",type=int,help="是否进行畸变矫正（全部），0->NO，1->YES",
                        default=0)
    parser.add_argument("--camMat",type=float,nargs='+',help="相机的内参矩阵,以00,11,02,22的顺序",
                        default=[1.46489094e+03,1.46358621e+03,3.53922761e+02,2.07792630e+02])
    parser.add_argument("--distCoeffs",type=float,nargs='+',help="畸变参数",
                        default=[-1.11169695e-01 ,1.34520353e+01,1.12320430e-02,7.68615307e-02,0])
    parser.add_argument("--doImageFusion", type=int, help="是否进行融合，0->NO，1->YES",
                        default=0)
    args = parser.parse_args()
    # 读取
    # 两张图片的情况
    if args.savepath[-4:-1]!=".png" or args.savepath[-4:-1]!=".jpg":
        raise Exception("Savepath must end with .png or .jpg")
    if len(args.list) < 2:
        raise Exception("Invalid Number Of Input Images")
    if len(args.list) == 2:
        img1 = args.list[0]
        img2 = args.list[1]
        stitchImage(img1,img2,imageSavePath=args.savepath,kpsAlgorithm=args.kpsAlgorithm,warpedSize=args.warpedSize,
                    gammaAlgorithm=args.gammaAlgorithm,gamma=args.gamma,adaptiveEqual=args.adaptiveEqual,
                    distorCorrect=args.distorCorrect,camMat=args.camMat,distCoeffs=args.distCoeffs,doImageFusion=args.doImageFusion)
        addText(args.savepath,args.list,args.savepath)
    # 多张图片的情况
    elif len(args.list) > 2:
        imglist = []
        for i in args.list:
            imglist.append(i)
        imgnums = len(imglist)
        img1 = imglist[0]
        img2 = imglist[1]
        stitchImage(img1,img2,imageSavePath=args.savepath,kpsAlgorithm=args.kpsAlgorithm,warpedSize=args.warpedSize,
                    gammaAlgorithm=args.gammaAlgorithm,gamma=args.gamma,adaptiveEqual=args.adaptiveEqual,
                    distorCorrect=args.distorCorrect,camMat=args.camMat,distCoeffs=args.distCoeffs,doImageFusion=args.doImageFusion)
        for i in range(3, imgnums + 1):
            imgi = imglist[i - 1]
            stitchImage(args.savepath,imgi,imageSavePath=args.savepath,kpsAlgorithm=args.kpsAlgorithm,warpedSize=args.warpedSize,
                        gammaAlgorithm=args.gammaAlgorithm,gamma=args.gamma,adaptiveEqual=args.adaptiveEqual,
                        distorCorrect=args.distorCorrect,camMat=args.camMat,distCoeffs=args.distCoeffs,doImageFusion=args.doImageFusion)
        addText(args.savepath, args.list, args.savepath)
    ############################################
import numpy as np
import cv2
import argparse

def detectAndDescribe(image,kpsAlgorithm=0):
    # SIFT特征点提取
    # 参数列表 （不全）
    # nfeatures 特征点数目 算法对检测出的特征点排名 返回最好的nfeatures个特征点 默认为0
    # contrastThreshold 过滤掉较差的特征点的对阈值 contrastThreshold越大 返回的特征点越少 默认0.04
    # edgeThreshold 过滤掉边缘效应的阈值 edgeThreshold越大，特征点越多（被过滤掉的越少）默认 10
    # sigma 金字塔第0层图像高斯滤波系数 默认1.6
    #descriptor = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    if kpsAlgorithm==0:
        descriptor = cv2.xfeatures2d.SURF_create()
    elif kpsAlgorithm==1:
        descriptor = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    elif kpsAlgorithm==2:
        descriptor = cv2.ORB_create()
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
    H[0, :] -= H[2, :] * min_out_w
    H[1, :] -= H[2, :] * min_out_h
    # 下面是计算投影图像的宽高
    # correct_w, correct_h = cv2.perspectiveTransform(corner_pts, H)[0].max(axis=0).astype(np.int_)
    # 返回平移的值
    #
    moveX = abs(min_out_w)
    moveY = abs(min_out_h)
    return H, moveX, moveY
# transformA参数
# imageAPath和imageBPath是图片的路径 传进来再读取
# path 图片的保存路径
# ratio 筛选好的匹配对的参数
# reprojThresh 计算单应矩阵需要的参数

# 先将第一张图片变换到第二张图片的坐标系下，保存仿射变换后的图像
# 第一次提取默认使用SURF算法
def transformImage(imageAPath,imageBPath,interImageSavePath='.\\transformed.png',transKpsAlgorithm=0,ratio=0.75, reprojThresh=4.0):
    # 读取两张待拼接的两张图片
    imageA = cv2.imread(imageAPath)
    imageB = cv2.imread(imageBPath)
    # 这里的情况只考虑两张拼接图像是相同size的
    h,w,c = imageA.shape
    # 通过detectAndDescribe函数得到关键点（这里的kpsA B是坐标）及其对应的descriptors描述子(featuresA B其实这里不应该写成特征的)
    (kpsA, featuresA) = detectAndDescribe(imageA,transKpsAlgorithm)
    (kpsB, featuresB) = detectAndDescribe(imageB,transKpsAlgorithm)
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
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0] + imageB.shape[0]))
    # 裁剪黑边
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    result = result[min_row:max_row, min_col:max_col, :]
    cv2.imwrite(interImageSavePath, result)

# 第二次提取特征点默认用ORB算法
def stitch(imageBPath,transformedImagePath='.\\transformed.png',resultImageSavePath=".\\result.png",stitchKpsAlgorithm=0,ratio=0.75, reprojThresh=4.0):
    # 读取
    transformed = cv2.imread(transformedImagePath)
    imageB = cv2.imread(imageBPath)
    # 加黑边，防止平移的时候移出边界
    imageH,imageW,imageC = transformed.shape
    top, bot, left, right = imageH, imageH, imageW, imageW
    transformed = cv2.copyMakeBorder(transformed, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # 给B加上黑边再去找匹配的特征点，不加会导致transformed图片在左上角去平移
    imageBTmp = cv2.copyMakeBorder(imageB, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # 找特征点 然后找平移的H
    (kpsT, featuresT) = detectAndDescribe(transformed,stitchKpsAlgorithm)
    (kpsB, featuresB) = detectAndDescribe(imageBTmp,stitchKpsAlgorithm)
    # 这里H计算不要计算反了  这里是将转换后的图像去进行平移 所以H是变换transformed的矩阵
    M = matchKeypoints(kpsB, kpsT, featuresB, featuresT,ratio,reprojThresh)
    if M is None:
        return None
    (matches, H, status) = M
    result = cv2.warpPerspective(transformed,H, (transformed.shape[1], transformed.shape[0]),
                                  flags=cv2.WARP_INVERSE_MAP)
    # 这里拷贝图片应该是以添加上、左黑边长度为起始点
    result[imageH:imageH+imageB.shape[0], imageW:imageW+imageB.shape[1]] = imageB
    # 裁剪黑边
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    result = result[min_row:max_row, min_col:max_col, :]
    cv2.imwrite(resultImageSavePath, result)

# 先不考虑拼接的顺序
# 用ORB进行检测，平移，效果明显不如SURF
if __name__ == '__main__':
    #文件路径
    # testImage1Path = "./data/Vis2048/test3/1.jpg"
    # testImage2Path = "./data/Vis2048/test3/2.jpg"
    # testImage3Path = "./data/Vis2048/test3/3.jpg"
    # testImage4Path = "./data/Vis2048/test3/4.jpg"
    # transformImage(testImage1Path,testImage2Path,interImageSavePath='.\\transformed.png')
    # stitch(testImage2Path, transformedImagePath='.\\transformed.png', resultImageSavePath=".\\result.png")
    # transformImage(".\\result.png", testImage3Path, interImageSavePath='.\\transformed.png')
    # stitch(testImage3Path, transformedImagePath='.\\transformed.png', resultImageSavePath=".\\result.png")
    # transformImage(".\\result.png", testImage4Path, interImageSavePath='.\\transformed.png')
    # stitch(testImage4Path, transformedImagePath='.\\transformed.png', resultImageSavePath=".\\result.png")
    ############################################
    # 参数设置   不考虑拼接的顺序情况
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", "-l", type=str, nargs='+', help="待拼接的图片路径")
    parser.add_argument("--transpath", "-tp", type=str, help="映射变换后的保存路径",
                        default=".\\transformed.png ")
    parser.add_argument("--transKpsAlgorithm",type=int,help="第一张图像映射变换时，特征点提取算法，默认是SURF",
                        default=0)
    parser.add_argument("--savepath", "-sp", type=str, help="最终结果保存路径",
                        default=".\\result.png")
    parser.add_argument("--stitchKpsAlgorithm", type=int, help="映射后图像进行平移时，特征点提取算法，默认是SURF",
                        default=0)
    args = parser.parse_args()
    # 读取
    # 两张图片的情况
    if len(args.list) == 2:
        img1 = args.list[0]
        img2 = args.list[1]
        transformImage(img1,img2,interImageSavePath=args.transpath,transKpsAlgorithm=args.transKpsAlgorithm)
        stitch(img2, transformedImagePath=args.transpath, resultImageSavePath=args.savepath,stitchKpsAlgorithm=args.stitchKpsAlgorithm)
    # 多张图片的情况
    elif len(args.list) > 2:
        imglist = []
        for i in args.list:
            imglist.append(i)
        imgnums = len(imglist)
        img1 = imglist[0]
        img2 = imglist[1]
        transformImage(img1, img2, interImageSavePath=args.transpath, transKpsAlgorithm=args.transKpsAlgorithm)
        stitch(img2, transformedImagePath=args.transpath, resultImageSavePath=args.savepath,stitchKpsAlgorithm=args.stitchKpsAlgorithm)
        for i in range(3, imgnums + 1):
            imgi = imglist[i - 1]
            transformImage(args.savepath, imgi, interImageSavePath=args.transpath, transKpsAlgorithm=args.transKpsAlgorithm)
            stitch(imgi, transformedImagePath=args.transpath, resultImageSavePath=args.savepath,stitchKpsAlgorithm=args.stitchKpsAlgorithm)
    ############################################
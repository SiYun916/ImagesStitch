import numpy as np
import cv2
import argparse
def detectAndDescribe(image,algorithm):
    # 将彩色图片转换成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 选择算法
    if algorithm==0:
        descriptor = cv2.SIFT_create()
        #descriptor = cv2.ORB_create()
    else:
        #descriptor = cv2.SIFT_create()
        descriptor = cv2.ORB_create()
    # SURF只有在opencv2==3.4.2.16等低版本时才能直接使用
    # descriptor = cv2.xfeatures2d.SURF_create()
    # descriptor = cv2.BRISK_create()
    # descriptor = cv2.ORB_create()
    # 检测特征点，并计算描述子
    (kps, features) = descriptor.detectAndCompute(image, None)
    # 将结果转换成NumPy数组
    kps = np.float32([kp.pt for kp in kps])
    # 返回特征点集，及对应的描述特征
    return (kps, features)
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio = 0.75, reprojThresh = 4.0):
    # 建立匹配器
    matcher = cv2.BFMatcher()
    # 使用KNN检测来自A、B图的特征匹配对，K=2
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
        # 计算单应矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        # 返回结果
        return (matches, H, status)
    # 如果匹配对小于4时，返回None
    return None
def correct_H(H,w,h):
    corner_pts = np.array([[[0, 0], [w, 0], [0, h], [w, h]]], dtype=np.float32)
    min_out_w, min_out_h = cv2.perspectiveTransform(corner_pts, H)[0].min(axis=0).astype(np.int_)
    H[0, :] -= H[2, :] * min_out_w
    H[1, :] -= H[2, :] * min_out_h
    correct_w, correct_h = cv2.perspectiveTransform(corner_pts, H)[0].max(axis=0).astype(np.int_)
    return H, correct_w, correct_h
def transformA(imageA,imageB,path='.\\transformed.png',ratio=0.75, reprojThresh=4.0,algorithm=0):
    image1 = cv2.imread(imageA)
    image2 = cv2.imread(imageB)
    h,w,c = image1.shape
    (kpsA, featuresA) = detectAndDescribe(image1,algorithm)
    (kpsB, featuresB) = detectAndDescribe(image2,algorithm)
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    if M is None:
        return None
    (matches, H, status) = M
    (H, correct_w, correct_h) = correct_H(H,w,h)
    result = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0] + image2.shape[0]))
    cv2.imwrite(path, result)
def stitch(imageB,tranformedA=".\\transformed.png",savepath=".\\processed.png",stitchborder=0):
    img1 = cv2.imread(imageB)
    img2 = cv2.imread(tranformedA)
    h,w,c = img1.shape
    top, bot, left, right = h+stitchborder, h+stitchborder, w+stitchborder, w+stitchborder
    srcImg = cv2.copyMakeBorder(img1, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv2.copyMakeBorder(img2, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img1gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    img2gray = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1gray, None)
    kp2, des2 = orb.detectAndCompute(img2gray, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, 2)
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
    #rows, cols = srcImg.shape[:2]
    MIN_MATCH_COUNT = 4
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        warpImg = cv2.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                     flags=cv2.WARP_INVERSE_MAP)
        warpImg[h+stitchborder:h+stitchborder+img1.shape[0],w+stitchborder:w+stitchborder+img1.shape[1]] = img1
        rows, cols = np.where(warpImg[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        warpImg = warpImg[min_row:max_row, min_col:max_col, :]
        cv2.imwrite(savepath, warpImg)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
def matchCount(featuresA, featuresB, ratio = 0.75):
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        return True
    else:
        return False
def judgeCom(imageA,imageB):
    image1 = cv2.imread(imageA)
    image2 = cv2.imread(imageB)
    (kpsA, featuresA) = detectAndDescribe(image1,algorithm=0)
    (kpsB, featuresB) = detectAndDescribe(image2,algorithm=0)
    Res = matchCount(featuresA, featuresB, 0.75)
    return Res
def getPairList(files,firstImg):
    lst = []
    for file in files:
        judgeRes=judgeCom(file,firstImg)
        if judgeRes:
            lst.append(1)
        else:
            lst.append(0)
    return lst
def getHLists(files):
    HLists = []
    fileNum = len(files)
    for i in range(0,fileNum-1):
        firstImg = files[0]
        files.remove(firstImg)
        listTmp = getPairList(files,firstImg)
        HLists.append(listTmp)
    return HLists
def sortFile(status,files):
    sortedFiles = []
    sortedFiles.append(files[0])
    statusQueue = []
    statusQueue.append(status[0])
    t = 0
    while len(statusQueue) != 0 :
        lsttmp = statusQueue[0]
        statusQueue.remove(lsttmp)
        for i in range(0,len(lsttmp)):
            if lsttmp[i] == 1:
                sortedFiles.append(files[t+i+1])
                if i+1 <len(status):
                    statusQueue.append(status[i+1])
                if len(sortedFiles) == len(files):
                    break
        if len(sortedFiles)==len(files):
            break
        t = t + 1
    return sortedFiles
def imageSort(imglist):
    files = []
    filesBackup = []
    for file in imglist:
        files.append(file)
        filesBackup.append(file)
    status = getHLists(files)
    newFileList = sortFile(status,filesBackup)
    return newFileList
if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", "-l", type=str, nargs='+', help="待拼接的图片路径")
    parser.add_argument("--transpath", "-tp", type=str, help="映射变换后的保存路径",
                        default=".\\transformed.png ")
    parser.add_argument("--savepath", "-sp", type=str, help="最终结果保存路径",
                        default=".\processed.png ")
    # parser.add_argument("--stitchborder", "-sb", type=int, help="拼接时为保证可以完成拼接而增添的边框",
    #                     default=0)
    args = parser.parse_args()
    # 读取
    if len(args.list) == 2:
        img1 = args.list[0]
        img2 = args.list[1]
        tmp = cv2.imread(img1)
        tmph,tmpw,tmpc = tmp.shape
        algo = 0
        if tmpw > 2500:
            algo = 1
        transformA(img1, img2,path=args.transpath,algorithm=algo)
        stitch(img2,tranformedA=args.transpath,savepath=args.savepath)
    elif len(args.list) > 2:
        imglist = []
        for i in args.list:
            imglist.append(i)
        tmp = cv2.imread(imglist[0])
        tmph, tmpw, tmpc = tmp.shape
        algo = 0
        if tmpw > 2500:
            filelist = imglist
            algo = 1
        elif tmpw < 2000:
            filelist = imageSort(imglist)
        else:
            filelist = imglist
        filenums = len(filelist)
        img1 = filelist[0]
        img2 = filelist[1]
        transformA(img1, img2,path=args.transpath,algorithm=algo)
        stitch(img2, tranformedA=args.transpath,savepath=args.savepath)
        for i in range(3, filenums + 1):
            imgi = filelist[i - 1]
            transformA(args.savepath, imgi,path=args.transpath,algorithm=algo)
            stitch(imgi,tranformedA=args.transpath,savepath=args.savepath)
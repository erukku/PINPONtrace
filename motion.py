import numpy as np
import cv2 as cv
import argparse
import csv

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='VID_20190718_184216.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

fmt = cv.VideoWriter_fourcc('m','p','4','v')

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(8,8))

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(detectShadows = False)
else:
    backSub = cv.createBackgroundSubtractorKNN(detectShadows = False)
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
#capture = cv.VideoCapture(0) #カメラで画像取得

writer = cv.VideoWriter('./after.mp4',fmt,240,(360,640))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
count = 0 #フレーム数
Co = 0 #サーブの区切りを判別する為の変数
al = [] #いったん入れる用
aL = [] #最終的なデータ


while True:
    ret, fram = capture.read()
    count += 1

    if fram is None:
        break

    frame = cv.cvtColor(fram, cv.COLOR_BGR2GRAY)
    ret, frame = cv.threshold(frame, 120, 255, cv.THRESH_BINARY)#二値化 120ちょうどいい

    fgMask = backSub.apply(frame)
    contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        
        counts=len(contours[i])
        area = cv.contourArea(contours[i]) #面積計算
        
        x=0.0
        y=0.0
        for j in range(counts):
            x+=contours[i][j][0][0]
            y+=contours[i][j][0][1]
                
        x/=counts
        y/=counts
        x=int(x)
        y=int(y)

        if x >= 1000:
            #print(10)
            continue

        if(area>=400):#面積が一定以下
            #print(x,y)
            cv.drawContours(fgMask,contours,i,(0,0,0),-1)#黄
        elif(area <= 20):
            cv.drawContours(fgMask,contours,i,(0,0,0),-1)#黄

    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    fgMask2 = cv.morphologyEx(fgMask, cv.MORPH_GRADIENT, kernel)

    size = (1920//3,1080//3)
    angle = -90

    scale = 1

    frame = cv.resize(frame,size)
    fgMask = cv.resize(fgMask,size)
    fgMask2 = cv.resize(fgMask2,size)

    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    fgMask = cv.rotate(fgMask, cv.ROTATE_90_CLOCKWISE)
    fgMask2 = cv.rotate(fgMask2, cv.ROTATE_90_CLOCKWISE)

    cir = cv.HoughCircles(fgMask,cv.HOUGH_GRADIENT,1,10000,
                            param1=200,param2=10,minRadius=2,maxRadius=7)
    cir2 = cv.HoughCircles(fgMask2,cv.HOUGH_GRADIENT,1,10000,
                            param1=200,param2=10,minRadius=2,maxRadius=7)                        
    kp = 0

    if not cir is None:
        for i in cir[0,:]:
            if i[1] <= 380:
                al.append([(i[0],i[1]),i[2]])
                if kp == 0:
                    kp = 1
    
    if not cir2 is None:
        for i in cir2[0,:]:
            if i[1]>=380:
                al.append([(i[0],i[1]),i[2],count])
                
                if not kp > 2:
                    kp = 2

    if kp == 0:
        if Co >= 80: #一定以上の間変化がない/区切る
            aL.append(al)
            al = []
        Co += 1
    else:
        Co = 0
          
    for i in al: #動画に玉の位置を出力する
        cv.circle(fram,(i[0][0],i[0][1]),i[1],(255,255,0),-1)

    cv.rectangle(frame, (950, 530), (970,570), (0,255,0), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255))
    if count == 799:
        for i in al:
            cv.circle(fram,(i[0][0],i[0][1]),i[1],(255,255,0),-1)

    cv.imwrite('work2.png', fram)
    img = cv.imread('work2.png')
    writer.write(img)
    print("-------"+ str(count) +"-----------"+str(fgMask.shape))

    #cv.imshow('Frame', fram) #実行時にウィンドウに表示
    #cv.imshow('FG Mask', fgMask)
    
    #keyboard = cv.waitKey(1)
    #if keyboard == 'q' or keyboard == 27: 終了為のキー
    #   break


#csv書き出し
csv_file = open('test.csv', 'w')
fieldnames = ['frame','Place_x','Place_y','radius', 'OnGround','times']
wRiter = csv.DictWriter(csv_file, fieldnames=fieldnames)
wRiter.writeheader()
co = 1
for i in range(len(aL)):
    Co = 0
    for j in range(len(aL[i])):
        if aL[i][j][0][1] >= 400 and len(aL[i]) >= 40:
            wRiter.writerow({'frame':aL[i][j][-1],'Place_x': aL[i][j][0][0], 'Place_y': aL[i][j][0][1],'radius':aL[i][j][1],'OnGround': 0,'times':co})
            Co += 1
    if Co >= 0:
        co += 1


import numpy as np
import cv2 as cv
import argparse

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

#writer = cv.VideoWriter('./outtest.mp4',fmt,10,(int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),int(capture.get(cv.CAP_PROP_FRAME_WIDTH))),True)
writer = cv.VideoWriter('./after.mp4',fmt,240,(360,640))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
count = 0
al = []
# frameが200スタートでも問題なさそう
while True:
    ret, frame = capture.read()
    count += 1
    if count < 0:
        continue
    elif count >= 800:
        #writer.release()
        cv.destroyAllWindows()
        exit()
        
    if frame is None:
        break
    
    #frame = cv.fastNlMeansDenoisingColored(frame,None,10,10,7,9);

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.fastNlMeansDenoising(frame,300)
    ret, frame = cv.threshold(frame, 120, 200, cv.THRESH_BINARY)#120
    fgMask = backSub.apply(frame)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

    contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
        
        counts=len(contours[i])
        area = cv.contourArea(contours[i])#面積計算
        x=0.0
        y=0.0
        for j in range(counts):
            x+=contours[i][j][0][0]
            y+=contours[i][j][0][1]
                
        x/=counts
        y/=counts
        x=int(x)
        y=int(y)
        
        if(area>=300):#面積が一定以下
            cv.drawContours(fgMask,contours,i,(0,0,0),1)#黄



    #size = (1080//3,1920//3)
    size = (1920//3,1080//3)
    angle = -90

    scale = 1
    """
    frame = cv.resize(frame,(1920,1080))
    fgMask = cv.resize(fgMask,(1920,1080))

    print((frame.shape[1]/2,frame.shape[0]/2))
    rotation_matrix = cv.getRotationMatrix2D((frame.shape[1]/2,frame.shape[0]/2), angle, scale)
    frame = cv.warpAffine(frame, rotation_matrix,(1920,1080), flags=cv.INTER_CUBIC)

    fgMask = cv.warpAffine(fgMask, rotation_matrix, (1920,1080), flags=cv.INTER_CUBIC)


    cv.rectangle(frame, (950, 530), (970,570), (255,255,255), -1)
    #cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
              # cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

"""
    frame = cv.resize(frame,size)
    fgMask = cv.resize(fgMask,size)

    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    fgMask = cv.rotate(fgMask, cv.ROTATE_90_CLOCKWISE)

    #frame = cv.fastNlMeansDenoising(frame,50)
    #fgMask = cv.fastNlMeansDenoising(fgMask,25)
    cir = cv.HoughCircles(fgMask,cv.HOUGH_GRADIENT,1,1000,
                            param1=150,param2=10,minRadius=2,maxRadius=7)
    if not cir is None:
        #cir = np.uint16(np.around(cir))
        print(1)
        for i in cir[0,:]:
            print(i[0],i[1])
            if i[2]<=7:
                al.append([(i[0],i[1]),i[2]])
            
    for i in al:
        cv.circle(fgMask,(i[0][0],i[0][1]),i[1],(255,255,0),-1)

    cv.rectangle(frame, (950, 530), (970,570), (0,255,0), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255))
    if count == 799:
        for i in al:
            cv.circle(fgMask,(i[0][0],i[0][1]),i[1],(255,255,0),-1)

    cv.imwrite('work2.png', fgMask)
    img = cv.imread('work2.png')
    writer.write(img)
    print("-------"+ str(count) +"-----------"+str(fgMask.shape))
    #cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break

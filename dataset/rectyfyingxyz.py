import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2


import matplotlib.pyplot as plt

#e = lux.getCameraPosition()
#e = (0.00045402414980344474, 0.30297407507896423, 0.30260199308395386)
#g = lux.getCameraLookAt()
#g =(0.0, 0.0, 0.0)
#t = lux.getCameraUp()
#t= (-5.2143551698691226e-08, 0.7071067690849304, -0.7071067690849304)
#f = lux.getCameraFocalLength()
#f = 49.99966049194336
path = "C:/Users/dotax/OneDrive/Desktop/keyshotdataset/final10089/"
def rectifyingimage(e,zxcvb,filename):
    pathofsave = "C:/Users/dotax/OneDrive/Desktop/keyshotdataset/recf4/"
    f = 2062.596
    g = (0.0, 0.0, 0.0)
    fov = 1
    dxzxc=np.tan(fov/2)*f*2
    print(512/dxzxc)
    g=e
    gtotal=np.sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2])
    g=(g[0]/gtotal,g[1]/gtotal,g[2]/gtotal)
    print("g",g)


    t=(g[0],0,g[2])
    img = cv2.imread(path+filename, cv2.IMREAD_UNCHANGED)
    img_counter = img.copy()
    img_transform = img.copy()
    ttotal=np.sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2])
    t=(t[0]/ttotal,t[1]/ttotal,t[2]/ttotal)
    cameraver = np.cross(g, t)
    #print(cameraver[1]*t[1]+cameraver[0]*t[0]+cameraver[2]*t[2])
    cameravertotal=np.sqrt(cameraver[0]*cameraver[0]+cameraver[1]*cameraver[1]+cameraver[2]*cameraver[2])
    cameraver=(cameraver[0]/cameravertotal,cameraver[1]/cameravertotal,cameraver[2]/cameravertotal)
    t = np.cross(cameraver,g )
    ttotal=np.sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2])
    t=(t[0]/ttotal,t[1]/ttotal,t[2]/ttotal)

    print("gXt",cameraver)
    print("t",t)
    print("e",e)
    print(g[2]*t[2]+g[0]*t[0]+g[1]*t[1])
    print(cameraver[2]*t[2]+cameraver[0]*t[0]+cameraver[1]*t[1])
    print(g[2]*cameraver[2]+g[0]*cameraver[0]+g[1]*cameraver[1])

    rtrans = np.matrix([[cameraver[0], cameraver[1], cameraver[2], 0], \
                        [t[0], t[1], t[2], 0], \
                        [-g[0], -g[1], -g[2],0], \
                        [0, 0, 0, 1]])
    ttrans = np.matrix([[1, 0, 0, -e[0]], \
                        [0, 1, 0, -e[1]], \
                        [0, 0, 1, -e[2]], \
                        [0, 0, 0, 1]])
    ctrans=rtrans*ttrans
    print("ctrans",ctrans)



    matrixofinternal1 = np.matrix([[f, 0, 0, 0], \
                                    [0, f, 0, 0], \
                                    [0, 0, 1, 0]])

    matrixofinternal2 = np.matrix([[14.22222017 , 0,  256], \
                                   [0, 14.22222017   , 256], \
                                   [0, 0,  1]])

    matrixofinternal = matrixofinternal2 * matrixofinternal1

    pointupleft = np.array([[-0.1, 0, 0.1, 1]]).T
    #pointupleft = np.array([[t[0]+e[0],t[1]+e[1],t[2]+e[2], 1]]).T
    #pointupright = np.array([[cameraver[0], cameraver[1], cameraver[2], 0]]).T
    pointupright = np.array([[0.1, 0, 0.1, 1]]).T
    pointbotleft =np.array([[-0.1, 0, -0.1, 1]]).T
    #pointbotleft =np.array([[g[0], g[1], g[2], 0]]).T
    pointbotright = np.array([[0.1, 0,-0.1, 1]]).T
    print("right",pointbotright)

    middle1=ctrans*pointupleft
    middle2=ctrans*pointupright
    middle3=ctrans*pointbotleft
    middle4=ctrans*pointbotright

    print("middle",middle1)
    print("2",middle2)
    print("3",middle3)
    print("444",middle4)


    xxx=matrixofinternal * middle1

    tupleft = (matrixofinternal2*matrixofinternal1 * middle1)/middle1[2]
    tupright = (matrixofinternal2*matrixofinternal1 * middle2)/middle2[2]
    botleft = (matrixofinternal2*matrixofinternal1 * middle3)/middle3[2]
    botright = (matrixofinternal2*matrixofinternal1 * middle4)/middle4[2]
    print(tupleft.reshape(-1))
    print(tupright)
    print(botleft)
    print(botright)
    tupleft[0]=512-tupleft[0]
    tupright[0]=512-tupright[0]
    botleft[0]=512-botleft[0]
    botright[0]=512-botright[0]
    tupleft[1]=512-tupleft[1]
    tupright[1]=512-tupright[1]
    botleft[1]=512-botleft[1]
    botright[1]=512-botright[1]
    #plotmatrix=(tupleft[0],tupright[0],botleft[0],botright[0],256)
    #plotmatrix2=(tupleft[1],tupright[1],botleft[1],botright[1],256)
    #plt.xlim((0, 512))
    #plt.ylim((0, 512))
    #plt.scatter(plotmatrix,plotmatrix2)

    #plt.show()
    srcPoints = np.vstack((tupleft[:2].reshape(-1), tupright[:2].reshape(-1), botleft[:2].reshape(-1), botright[:2].reshape(-1)))
    srcPoints = np.float32(srcPoints)
    print(srcPoints)


    long = 512
    short = 512

    canvasPoints = np.array([[0, 0], [int(long), 0], [0, int(short)], [int(long), int(short)]])
    canvasPoints = np.float32(canvasPoints)
    perspectiveMatrix = cv2.getPerspectiveTransform(srcPoints, canvasPoints)

    perspectiveImg = cv2.warpPerspective(img_transform, perspectiveMatrix,
                                         (int(long), int(short)))  # img_transform是原始图片，(int(long), int(short))画布尺寸
    #cv2.namedWindow("perspectiveImg", cv2.WINDOW_FREERATIO)
    #cv2.imshow("perspectiveImg", perspectiveImg)
    cv2.imwrite(pathofsave+filename, perspectiveImg)
    #cv2.waitKey(0)

import os


def chagestrtovector(str):
    ssss = str.split(',')
    strf=[]
    str1 = ssss[0].replace("(", "").replace(")", "")
    str2= ssss[1].replace("(", "").replace(")", "")
    str3 = ssss[2].replace("(", "").replace(")", "")
    strf=(float(str1),float(str2),float(str3))
    return(strf)



files= os.listdir(path)
s = []
for file in files: #
    if not os.path.isdir(file):
        keywordlist=file.split('_')
        e=chagestrtovector(keywordlist[3])
        #t=chagestrtovector(keywordlist[2])
        t=1
        rectifyingimage(e,t,file)
        #print(keywordlist,t,e)




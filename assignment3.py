
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


# Transformation of a RGB image to a greyscale image using the average method.
def T1(image):
    fill = 0
    bilist = []
    while fill < len(image):
        bilist.append([])
        fill += 1
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            (r, g, b) = image[i][j]
            x = (r/3)+(g/3)+(b/3)
            bilist[i].append(x)

    return np.array(bilist)


# This function generates histograms for both RGB and greyscale images.
def makeHist(img):
    if len(img.shape) == 3:
        row, col, third = img.shape
        size = row*col
        y1 = np.zeros(256)
        y2 = np.zeros(256)
        y3 = np.zeros(256)
        for i in range(0, row):
            for j in range(0, col):
                (r, g, b) = img[i][j]
                y1[r] += 1
                y2[g] += 1
                y3[b] += 1
        x1 = np.arange(len(y1))
        x2 = np.arange(len(y2))
        x3 = np.arange(len(y3))
        y1 = list(map(lambda x: x/size, y1))
        y2 = list(map(lambda x: x/size, y2))
        y3 = list(map(lambda x: x/size, y3))

        plt.bar(x1, y1, color="red", align="center")
        plt.bar(x2, y2, color="green", align="center")
        plt.bar(x3, y3, color="blue", align="center")
        plt.show()
        return y1, y2, y3
    elif len(img.shape) == 2:
        row, col = img.shape
        size = row*col
        y = np.zeros(256)
        for i in range(0, row):
            for j in range(0, col):
                y[img[i, j]] += 1
        y = list(map(lambda x: x/size, y))
        x = np.arange(0, 256)
        plt.bar(x, y, color="black", align="center")
        plt.show()
        return y


# this method call the makeHist method and will calculate s for each pixel and place it into a dictionary along with its index
def histequal(img):
    z = makeHist(img)
    p = 0
    l = 255
    dic = {}
    for i in range(0, len(z)):
        p += z[i]
        s = l * p
        dic[i] = round(s)

    return dic

# this method  call the histequal method and will iterate over each pixel in the image and insert the values from the dictionary into the new image.


def equalizeimg(img):
    fill = 0
    bilist = []
    while fill < len(img):
        bilist.append([])
        fill += 1
    b = histequal(img)
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            x = img[i][j]
            bilist[i].append(b[x])
    return np.array(bilist)

# end of assignment


def img_pool(img, pool):
    readimg = np.array(img)
    # leng = int(readimg.shape[0]/2)
    # wid = int(readimg.shape[1]/2)
    rtnimg = np.zeros((int(readimg.shape[0]/2), int(readimg.shape[1]/2)))
    for i in range(0, len(readimg)-1, 2):
        for j in range(0, len(readimg[i])-1, 2):
            x1 = readimg[i][j]
            x2 = readimg[i][j+1]
            x3 = readimg[i+1][j]
            x4 = readimg[i+1][j+1]
            lst = [x1, x2, x3, x4]
            # zero for max pool, 1 for average pool
            if pool == 0:
                rtnimg[int(i/2)][int(j/2)] = max(lst)
            elif pool == 1:
                rtnimg[int(i/2)][int(j/2)] = round(sum(lst)/len(lst))
            else:
                sys.exit("Enter a valid pool. 1 = max, 2 = avg")
    return np.array(rtnimg)



def add_rgb(img1, img2):

    return 0






image = cv2.imread("CIS_465\html\images\jWebb.jpg")
resize = cv2.resize(image, [800, 600], interpolation=cv2.INTER_AREA)

grey = T1(resize)
grey = grey.astype(np.uint8)

output = img_pool(grey, 1)
output2 = img_pool(grey, 0)
output = output.astype(np.uint8)


makeHist(resize)
makeHist(grey)

cv2.imshow('Max Pool', resize0)
cv2.imshow('Avg Pool', output2.astype(np.uint8))
print("press 0 to quit.")


import imutils
import numpy as np
import cv2
from PIL import Image
import glob
#this is a code
#Detect the skin area from an image
def detect_skin(frame,lower,upper):
    frame = imutils.resize(frame, width=400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=1)
    skinMask = cv2.dilate(skinMask, kernel, iterations=1)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    return frame,skin

# Currently it is not needed
def removeBG(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    c = find_biggest_contours(frame)
    rectangle = cv2.boundingRect(c)
    mask = np.zeros(frame.shape[:2], np.uint8)
    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Run grabCut
    cv2.grabCut(frame,  # Our image
                mask,  # The Mask
                rectangle,  # Our rectangle
                bgdModel,  # Temporary array for background
                fgdModel,  # Temporary array for background
                5,  # Number of iterations
                cv2.GC_INIT_WITH_RECT)  # Initiative using our rectangle
    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # Multiply image with new mask to subtract background
    image_rgb_nobg = frame * mask_2[:, :, np.newaxis]
    return image_rgb_nobg

#Find the biggest contours to extract the hand
def find_biggest_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    '''
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(image, cnts, i, (255, 0, 0), 3)
    cv2.imshow('cnt', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    c = max(cnts, key=cv2.contourArea)
    return c

lower = np.array([0, 48 , 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
image_list = []
folder_list = []
all_alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


for filename in glob.glob('input/A/*.jpg'):
    im = Image.open(filename)
    image_list.append(im)

for i in range(len(image_list)):
    I = cv2.cvtColor(np.array(image_list[i]), cv2.COLOR_BGR2RGB)
    #cv2.imshow('asd',np.array(I))
    resized_frame, skin = detect_skin(I, lower, upper)
    c = find_biggest_contours(skin)
    x, y, width, height = cv2.boundingRect(c)
    only_hand = skin[y:y + height, x:x + width]
    #cv2.imshow("images", np.hstack([resized_frame, skin]))
    #cv2.imshow("extracted", only_hand)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    path = 'processed_output/A/' +str(i) + '.jpg'
    cv2.imwrite(path, only_hand)
'''
frame = cv2.imread('/home/turzo/PycharmProjects/Image_processing_project_1/dataset/Dataset/user_3/A0.jpg')
lower = np.array([0, 48 , 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
resized_frame,skin = detect_skin(frame,lower,upper)
#removed_background = removeBG(resized_frame)
c = find_biggest_contours(skin)
x,y,width,height = cv2.boundingRect(c)
only_hand = skin[y:y+height,x:x+width]
cv2.imshow("images", np.hstack([resized_frame, skin]))
cv2.imshow("extracted",only_hand)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
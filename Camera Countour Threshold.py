import cv2
import numpy as np


cap = cv2.VideoCapture(0)

#Resizes the overlay rectangle to best fit the resolution of the camera
#New sizes are also used for masking the image later on.
ret, frame = cap.read()
size = frame.shape[::-1]
w = size[1]
h = size[2]

card_w = 750
card_h = 1050

#Scales the card size down or up to best fit the camera in use.
if card_h >= h:
    while card_h >= h:
        card_h = card_h * .90
        card_w = card_w * .90
    print("Card size decreased to: " + str(card_h) + "*" + str(card_w))
else:
    while card_h <= h:
        card_h = card_h * 1.1
        card_w = card_w * 1.1
    card_h = card_h * .90
    card_w = card_w * .90
    print("Card size increased to: " + str(card_h) + "*" + str(card_w))

#Sets the cordinates for the rectangle overlay and mask rectangle.
rect_top_x = int((w - card_w) / 2)
rect_top_y = int((h - card_h) / 2)
rect_bottom_x = int(rect_top_x + card_w)
rect_bottom_y = int(rect_top_y + card_h)

#Camera feed, terminates when pressing 'q'.
#Also captures screenshot when terminated for analysis.
while True:
    ret, frame = cap.read()
    size = frame.shape[::-1]

    cv2.rectangle(frame, (rect_top_x, rect_top_y), (rect_bottom_x, rect_bottom_y), (255,255,255))
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite("CardImg.png", frame)
        break

#Reads in screenshot, creates arrays for later use with masking.
img = cv2.imread("CardImg.png")
mask = np.zeros(img.shape[:2], np.uint8)

back = np.zeros((1, 65), np.float64)
front = np.zeros((1, 65), np.float64)

#Sets rectangle cords for masking foreground
rect = (rect_top_x, rect_top_y, int(card_w), int(card_h))

#Creates mask, then converts mask and img into arrays, multiplies them to
#"crop" the masked portion.
cv2.grabCut(img, mask, rect, back, front, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1)
img = img*mask2[:, :, np.newaxis]

#Reformats the masked img array back into a img.
cv2.imwrite("mask.png", img)
img = cv2.imread("mask.png", 1)
cv2.imshow("Img", img)

#Applies addaptive gaussian filter to masked image to attempt to better real lettering.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 191, 1)

cv2.imshow("Adaptive Threshold", adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()
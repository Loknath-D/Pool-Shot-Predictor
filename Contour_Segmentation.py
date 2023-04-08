import cv2
import numpy as np
import cvzone

class Contour_Segmentor:
    def __init__(self):
        pass

    def segment(self, img, hsv_range1 = [], hsv_range2 = [], min_area = 100, draw = True):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV);
        low_hsv = np.array(hsv_range1);
        high_hsv = np.array(hsv_range2);
        mask = cv2.inRange(hsv, low_hsv, high_hsv);
        imgContours, contours = cvzone.findContours(img, mask, minArea = min_area);
        if (draw):
            if (contours):
                cv2.drawContours(image = img, contours = contours[0]['cnt'],
                                 contourIdx = -1, color = (0, 255, 0),
                                 thickness = 2, lineType = cv2.LINE_AA);
        return contours;


import cv2 as cv
import numpy as np
import imutils

def main():
    real = cv.imread("images/pcb-real-3.png", cv.IMREAD_COLOR)
    cad = cv.imread("images/pcb-cad-3.png", cv.IMREAD_GRAYSCALE)

    cad = cv.resize(cad, (real.shape[1], real.shape[0]))

    real = cv.GaussianBlur(real, (11,11), 0)
    hsv = cv.cvtColor(real, cv.COLOR_BGR2HSV)

    # mask1 = cv.inRange(hsv, (170, 38, 38), (180, 153, 255))
    mask2 = cv.inRange(hsv, (8,  20, 10), (40,  255,  127))
    mask2 = cv.bitwise_not(mask2)

    cv.imshow("mask", mask2)
    cv.waitKey()

    real = cv.cvtColor(real, cv.COLOR_BGR2GRAY)
    # real = cv.bitwise_and(real, mask2)

    # (_, real) = cv.threshold(real, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    (_, cad) = cv.threshold(cad, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    real = cv.adaptiveThreshold(real, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 301, 0)
    # cad = cv.adaptiveThreshold(cad, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 0)

    diff = cv.bitwise_xor(real, cad)

    cv.imshow("real", real)
    cv.imshow("cad", cad)
    cv.imshow("diff", diff)
    cv.waitKey()

    # real = cv.findContours(real, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # real = imutils.grab_contours(real)
    

if __name__ == "__main__":
    main()

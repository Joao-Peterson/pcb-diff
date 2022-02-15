import cv2 as cv
import numpy as np
import imutils

def main():
    orig = cv.imread("images/pcb-1-fail.png", cv.IMREAD_COLOR)
    cad = cv.imread("images/pcb-1-mask.png", cv.IMREAD_GRAYSCALE)

    cad = cv.resize(cad, (orig.shape[1], orig.shape[0]))

    real = cv.GaussianBlur(orig, (11,11), 0)
    cad = cv.GaussianBlur(cad, (11,11), 0)

    real = cv.cvtColor(real, cv.COLOR_BGR2GRAY)

    # (_, real) = cv.threshold(real, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    (_, cad) = cv.threshold(cad, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    real = cv.adaptiveThreshold(real, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 301, 0)
    # cad = cv.adaptiveThreshold(cad, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 0)

    diff = cv.bitwise_xor(real, cad)
    diff = cv.GaussianBlur(diff, (31,31), 0)
    (_, diff) = cv.threshold(diff, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # cont = cv.findContours(diff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cont = imutils.grab_contours(cont)
    # for c in cont:
    #     (x, y, w, h) = cv.boundingRect(c)
    #     cv.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cont, hierarchy = cv.findContours(diff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(cont)):
        cv.drawContours(orig, cont, i, (0,0,255), 2, cv.LINE_8, hierarchy, 0)


    cv.imshow("orig", orig)
    # cv.imshow("real", real)
    # cv.imshow("cad", cad)
    cv.imshow("diff", diff)
    cv.waitKey()


if __name__ == "__main__":
    main()

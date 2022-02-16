from numpy import full, uint8
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2


cad = cv2.imread("images\pcb-1.png", cv2.IMREAD_UNCHANGED)
board = cv2.imread("images\pcb-1-fail.png", cv2.IMREAD_UNCHANGED)

pcb = cv2.resize(cad, (board.shape[1], board.shape[0]))

cv2.imshow("imageA", cad)
cv2.imshow("imageB", board)
cv2.waitKey(0) 

grayA = cv2.cvtColor(cad, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

""" cv2.imshow("grayA", grayA)
cv2.imshow("grayB", grayB)
cv2.waitKey(0) """

(score, diff) = compare_ssim(grayA, grayB, full = True)
diff = (diff * 255).astype(uint8)

""" cv2.imshow("diff", diff)
cv2.waitKey(0) """

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)


for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(pcb, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(board, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
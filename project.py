import cv2
import imutils
import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract\tesseract'

img = cv2.imread('pictures/plane.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('pictures/plane2.png', cv2.IMREAD_COLOR)

# test_img = cv2.imread('pictures/alfa-romeo.jpg', cv2.IMREAD_COLOR)
# test_text = pytesseract.image_to_string(test_img, config='--psm 11')
#
# print("This is the test text: " + test_text)

cropped_image = img[200:500, 700:1100]
cropped_image2 = img2[100:350, 400:800]

img_resized = cv2.resize(img2, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
img_resized_cropped = cv2.resize(cropped_image2, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
img_gray_cropped = cv2.cvtColor(img_resized_cropped, cv2.COLOR_BGR2GRAY)

kernel = np.ones((1, 1), np.uint8)
img_test = cv2.dilate(img_gray, kernel, iterations=1)
img_test = cv2.erode(img_test, kernel, iterations=1)

img_test_blur = cv2.threshold(cv2.medianBlur(img_test, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img_test_blur2 = cv2.threshold(cv2.bilateralFilter(img_test, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

img_test_threshold = cv2.adaptiveThreshold(cv2.medianBlur(img_test, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
img_test_threshold2 = cv2.adaptiveThreshold(cv2.bilateralFilter(img_test, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


# cv2.imshow('Image resized', img_resized)
# cv2.imshow('Image gray', img_gray)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(cropped_image2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
threshold_img2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# text2 = pytesseract.image_to_string(img2, config='--psm 11')
# text_gray = pytesseract.image_to_string(gray, config='--psm 11')
# text_gray2 = pytesseract.image_to_string(gray2, config='--psm 11')

# print("Detected text:", text, "Text2", text2, "Gray", text_gray, "Gray2", text_gray2)

plane_images = [img, cropped_image, threshold_img, img2, cropped_image2, gray2, threshold_img2, img_resized, img_gray]
plane_images_test = [img_resized, img_resized_cropped, img_gray, img_gray_cropped, img_test, img_test_blur, img_test_threshold, img_test_blur2, img_test_threshold2]

for x in plane_images_test:
    text = pytesseract.image_to_string(x, config='--psm 11 --oem 1 -l eng')
    # plane_number = re.search("[A-Z]{2}-[A-Z]{3}", text)
    # if plane_number == None:
    #     print("No plane number found")
    # else:
    #     print(plane_number.group())
    print(text)
    print("///////////////////////////////////////////////////////////////////")

# expected results plane1 LY-BGS, plane2 LZ-GNK

# cv2.imshow('Plane', cropped_image)
# cv2.imshow('Plane2', cropped_image2)
# cv2.imshow('Gray', gray)
# cv2.imshow('Gray2', gray2)
cv2.waitKey(0)

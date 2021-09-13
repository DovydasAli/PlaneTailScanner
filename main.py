import numpy as np
import re
import cv2
import pytesseract
from PIL import Image
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract\tesseract'
# print(pytesseract.image_to_string(r'C:\Users\Ketaminas\Downloads\other\plane.png'))
#
# img = Image.open('plane.png').crop((800,0,1000,400))
# img.show()

# # get grayscale image
# def get_grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# def thresholding(image):
#     return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# def canny(image):
#     return cv2.Canny(image, 100, 200)
#
# cv2.imwrite('planegray.png',get_grayscale(img))

# h, w, c = img.shape
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)



# d = pytesseract.image_to_string(img)
# print(d)
# keys = list(d.keys())
#
#
# date_pattern = 'LY-BGS'
#
# n_boxes = len(d['text'])
# for i in range(n_boxes):
#     if int(d['conf'][i]) > 60:
#     	if re.match(date_pattern, d['text'][i]):
# 	        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
# 	        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import pytesseract
root = Tk()
root.title('TechVidvan Text from image project')
newline= Label(root)
uploaded_img=Label(root)
scrollbar = Scrollbar(root)
scrollbar.pack( side = RIGHT, fill = Y )
def extract(path):
    Actual_image = cv2.imread(path)
    Sample_img = cv2.resize(Actual_image,(400,350))
    Image_ht,Image_wd,Image_thickness = Sample_img.shape
    Sample_img = cv2.cvtColor(Sample_img,cv2.COLOR_BGR2RGB)
    texts = pytesseract.image_to_data(Sample_img)
    print(texts)
    mytext=""
    prevy=0
    for cnt,text in enumerate(texts.splitlines()):
        if cnt==0:
            continue
        text = text.split()
        if len(text)==12:
            x,y,w,h = int(text[6]),int(text[7]),int(text[8]),int(text[9])
            if(len(mytext)==0):
                prey=y
            if(prevy-y>=10 or y-prevy>=10):
                print(mytext)
                Label(root,text=mytext,font=('Times',15,'bold')).pack()
                mytext=""
            mytext = mytext + text[11]+" "
            prevy=y
    Label(root,text=mytext,font=('Times',15,'bold')).pack()
    print(mytext)
def show_extract_button(path):
    extractBtn= Button(root,text="Extract text",command=lambda: extract(path),bg="#2f2f77",fg="gray",pady=15,padx=15,font=('Times',15,'bold'))
    extractBtn.pack()
def upload():
    try:
        path=filedialog.askopenfilename()
        image=Image.open(path)
        img=ImageTk.PhotoImage(image)
        uploaded_img.configure(image=img)
        uploaded_img.image=img
        show_extract_button(path)
    except:
        pass
uploadbtn = Button(root,text="Upload an image",command=upload,bg="#2f2f77",fg="gray",height=2,width=20,font=('Times',15,'bold')).pack()
newline.configure(text='\n')
newline.pack()
uploaded_img.pack()
root.mainloop()




import cv2
import os
import pytesseract
from constants import path_debug

#detects the name of the card by OCR (Optical Character Recognition)
def find_card_name(im, debug):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    #crop "name zone" of Pokemon card (upper-left corner)
    h,w=im.shape[:2]
    name_card=im[:int(h/8),:int(w/2)]
    if debug:
        cv2.imwrite(os.path.join(path_debug,"card_name.png"),name_card)

    #Threshold at 127 (text should be black - doesn't work for Darkness types whose name is white)
    name_card=cv2.cvtColor(name_card,cv2.COLOR_BGR2GRAY)
    _,im_th=cv2.threshold(name_card,127,255,cv2.THRESH_BINARY_INV)
    if debug:
        cv2.imwrite(os.path.join(path_debug,"card_name_th.png"),name_card)

    #ocr function
    text = pytesseract.image_to_string(im_th)

    return text

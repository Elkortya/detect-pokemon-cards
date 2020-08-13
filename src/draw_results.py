import cv2
import os

def draw_results(im,box,pkmn_name):
    h, w = im.shape[:2]

    col_bg=(0,0,0)
    col_text = (0, 255, 111)

    cv2.drawContours(im, [box], -1, col_text, 10)


    y0=int(h*0.85)
    y1=int(h*0.95)
    x0=int(w*0.1)
    x1=int(w*0.9)
    cv2.rectangle(im,(x0,y0),(x1,y1),col_bg,thickness=-1)

    text="This is a " + pkmn_name
    font_face=cv2.FONT_HERSHEY_SIMPLEX
    font_scale=w / 500
    text_thickness=15
    # a,b=cv2.getTextSize(text, font_face, font_scale, text_thickness)

    cv2.putText(im, text , (x0+int(w*0.05),int(y1*0.97)), fontFace=font_face,
                fontScale=font_scale, color=col_text, thickness=text_thickness)

    return im
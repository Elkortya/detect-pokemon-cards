"""
Find position of card in the image, crop, rotate and isolate that card
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from constants import path_debug



def rotateImage(image, angle, nocrop=False):
    """ Rotates an image

    :param image: (npa) image being processed
    :param angle: (float) angle of rotation (radians)
    :param nocrop: (bool) toggles the cropping of the rotated image if it doesn't fit in original size
    :return: (npa) roatated image
    """

    dt = image.dtype

    # handling case of color and grayscale images
    if len(image.shape) == 2:
        color = False
    elif len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    else:
        raise Exception("Incorrect image shape for rotateImage")
    h = image.shape[0]
    w = image.shape[1]

    # compute the rotation matrix based on the desired angle
    if nocrop:
        diag = np.sqrt(w ** 2 + h ** 2).astype(np.uint64)
        if color:
            img2 = np.zeros((diag, diag, 3))
        else:
            img2 = np.zeros((diag, diag))
        img2[int(diag / 2 - h / 2):int(diag / 2 + h / 2), int(diag / 2 - w / 2):int(diag / 2 + w / 2)] = image
        image = img2.copy()
        resulting_shape = (diag, diag)
        rot_mat = cv2.getRotationMatrix2D((int(diag / 2), int(diag / 2)), angle * 180 / np.pi, 1.0)
    else:
        resulting_shape = (w, h)
        image_center = tuple(np.array([w, h]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle * 180 / np.pi, 1.0)

    # apply rotation matrix
    if color:
        result0 = cv2.warpAffine(image[:, :, 0], rot_mat, resulting_shape, flags=cv2.INTER_LINEAR)
        result1 = cv2.warpAffine(image[:, :, 1], rot_mat, resulting_shape, flags=cv2.INTER_LINEAR)
        result2 = cv2.warpAffine(image[:, :, 2], rot_mat, resulting_shape, flags=cv2.INTER_LINEAR)
        result = np.zeros(image.shape)
        result[:, :, 0] = result0
        result[:, :, 1] = result1
        result[:, :, 2] = result2
    else:
        result = cv2.warpAffine(image, rot_mat, resulting_shape, flags=cv2.INTER_LINEAR)
    return result.astype(dt)


def imgradient(img, sobel):
    """Get image gradient by Sobel

    :param img:  (npa) image being processed
    :param sobel:  (int) kernel size of Sobel operator
    :return: (npa) Result of Sobel filter on image
    """

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel)
    return np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))


def distance(pt1, pt2):
    """Measure euclidian distance

    :param pt1: ([float]) coordinates of point 1
    :param pt2: ([float]) "" point 2
    :return: (float) euclidian distance between point 1 and point 2
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def find_card_in_img(im, debug):
    """Find position of card in the image, crop, rotate and isolate that card

    :param im: (npa) image being processed
    :param debug:  (bool) toggles saving intermediary images for debug purposes
    :return img_card_isol: (npa) image of card isolated from rest of the original image
    :return box: ([float]) coordinates of the 4 edge points of the card in the original image
    """
    global path_debug

    im_shape = im.shape[:2]
    im_area = im_shape[0] * im_shape[1]

    # convert color to gray and
    im_orig = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # slightly blur image to reduce noise
    im = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=2)

    # get gradient
    im_grad = imgradient(im, 3).astype(np.uint8)

    # threshold the gradient image
    _, im_grad_th = cv2.threshold(im_grad, 15, 255, cv2.THRESH_OTSU)
    im_grad_th = im_grad_th.astype(np.uint8)

    # morphological closing to take out small elements
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    im_grad_th = cv2.morphologyEx(im_grad_th, cv2.MORPH_CLOSE, kernel)
    im_grad_th_color = cv2.cvtColor(im_grad_th, cv2.COLOR_GRAY2BGR)  # debug

    # find contours
    min_shape = 0.05 * im_area  # min area a contour must have to be considered a potential card
    real_card_ratio = 0.72  # actual dimension ratio of a card (6.4 / 8.9 centimeters)

    contours, _ = cv2.findContours(im_grad_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    im_contour_candidates = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int)  # bbx for easier drawing
        area_box = cv2.contourArea(box)

        # Check 1 for "is this contours a card ?" : is the contour big enough
        if area_box >= min_shape:

            # debug
            cv2.drawContours(im_grad_th_color, [box], -1, (0, 0, 255),
                             2)  # on peut drawer le bounding box avec drawContours

            # Check 2 : is the dimension ratio correct ?
            L1 = distance(box[0], box[1])
            L2 = distance(box[1], box[2])
            if (min(L1 / L2, L2 / L1) >= real_card_ratio - 0.05) and min(L1 / L2, L2 / L1) <= real_card_ratio + 0.05:
                # print(L1/L2)
                cv2.drawContours(im_grad_th_color, [box], -1, (255, 0, 255),
                                 2)  # on peut drawer le bounding box avec drawContours
                # draw all candidate contours
                cv2.drawContours(im_contour_candidates, [box], -1, 255,
                                 2)  # on peut drawer le bounding box avec drawContours

    if debug:
        cv2.imwrite(os.path.join(path_debug, "contour_candidates.png"), im_contour_candidates)
        cv2.imwrite(os.path.join(path_debug, "contours_debug.png"), im_grad_th_color)

    # we then find contours another time from the bbox image, to get correct external contours (watch out for inner border)
    if len(contours) >= 2:
        contours, _ = cv2.findContours(im_contour_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

    im_contour = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
    if len(contours) == 1:
        rect = cv2.minAreaRect(contours[0])
        angle = rect[-1]
        box = cv2.boxPoints(rect).astype(np.int)

    # if we still have several possible contours, take the one with the ratio closest to the real ratio.
    if len(contours) >= 2:
        im_contour = np.zeros((im.shape[0], im.shape[1])).astype(np.uint8)
        ratios = []
        angles = []
        boxes = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            angles.append(rect[-1])
            box = cv2.boxPoints(rect).astype(np.int)

            if len(contours) >= 2:
                L1 = distance(box[0], box[1])
                L2 = distance(box[1], box[2])
                if (L1 >= 0.67 * L2 and L1 <= 0.77 * L2) or (
                        L2 >= 0.67 * L1 and L2 <= 0.77 * L1):  # ratio officiel 0.72
                    ratios.append(min(L1 / L2, L2 / L1))
                    boxes.append(box)

        # get the closest contour
        closest_index = np.argmax(abs(ratios - 0.72))
        contours = contours[closest_index]
        angle = angles[closest_index]
        box = boxes[closest_index]

    cv2.drawContours(im_contour, [box], -1, 255, 2)  # on peut drawer le bounding box avec drawContours
    if debug:
        cv2.imwrite(os.path.join(path_debug, "contour.png"), im_contour)

    # create mask from contour
    cv2.fillPoly(im_contour, pts=contours, color=255)
    # apply mask to original image
    im_card = cv2.bitwise_and(im_orig, im_orig, mask=im_contour)
    if debug:
        cv2.imwrite(os.path.join(path_debug, "detected_card.png"), im_card)

    # rotate card by angle found thanks to rect and contour
    rotated_im = rotateImage(im_card, +(angle + 90) / 180 * np.pi, nocrop=True)
    if debug:
        cv2.imwrite(os.path.join(path_debug, "rotated_card.png"), rotated_im)

    # crop image
    rotated_mask = rotateImage(im_contour, +(angle + 90) / 180 * np.pi, nocrop=True)
    contour, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    rect = cv2.boundingRect(contour[0])

    im_card_isol = rotated_im[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    if debug:
        cv2.imwrite(os.path.join(path_debug, "isolated_card.png"), im_card_isol)

    return im_card_isol, box

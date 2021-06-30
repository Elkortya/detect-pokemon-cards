import os
import cv2
import shutil

from detect_card import find_card_in_img
from detect_name import find_card_name
from fuzzy_match_string import match_name_to_pkmn_name
from draw_results import draw_results
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('-I', type=str, required=True,
                        help='image to process')
    parser.add_argument('-debug', type=bool, default=False,
                        help='debug mode')

    args = parser.parse_args()

    # Setting paths
    path_project = os.path.dirname(os.path.dirname(__file__))
    path_imgs = os.path.join(path_project,"imgs")
    path_res = os.path.join(path_project,"ressources")
    path_result = os.path.join(path_project,"result")

    # if debug, intermediary images will be saved in a debug folder
    if args.debug:
        path_debug = "../debug"
        shutil.rmtree("../debug")
        os.makedirs("../debug")

    # Image to process
    img_name = args.I

    img_path = os.path.join(path_imgs, img_name)

    im = cv2.imread(img_path)

    # Step 1 : detect the card position in the image
    im_card, box = find_card_in_img(im, args.debug)

    # Step 2 : detect the name in the card
    card_name = find_card_name(im_card, args.debug)

    # Step 3 : match name to the closest true Pokemon name
    pkmn_name = match_name_to_pkmn_name(card_name, path_res)

    # Step 4 : draw results
    result = draw_results(im, box, pkmn_name)
    cv2.imwrite(os.path.join(path_result, "result.png"), im)

    print("In this image, we believe this card is a ", pkmn_name)

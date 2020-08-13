import os
import cv2
import shutil

from detect_card import find_card_in_img
from detect_name import find_card_name
from fuzzy_match_string import match_name_to_pkmn_name
from draw_results import draw_results

if __name__ == "__main__":

    # Setting paths
    path_imgs = "../imgs"
    path_res = "../ressources"
    path_result = "../result"

    # if debug, intermediary images will be saved in a debug folder
    debug = False
    if debug:
        path_debug = "../debug"
        shutil.rmtree("../debug")
        os.makedirs("../debug")

    # Image to process
    # img_name="slowpoke.jpg"
    img_name = "elec.jpg"

    # Opening image
    img_path = os.path.join(path_imgs, img_name)
    im = cv2.imread(img_path)

    # Step 1 : detect the card position in the image
    im_card, box = find_card_in_img(im, debug)

    # Step 2 : detect the name in the card
    card_name = find_card_name(im_card, debug)

    # Step 3 : match name to the closest true Pokemon name
    pkmn_name = match_name_to_pkmn_name(card_name, path_res)

    # Step 4 : draw results
    result = draw_results(im, box, pkmn_name)
    cv2.imwrite(os.path.join(path_result, "result.png"), im)

    print("In this image, we believe this card is a ", pkmn_name)

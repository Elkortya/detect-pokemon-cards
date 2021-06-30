# Match the detected name to the closest one existing in the name database, according to levenshtein distance

import numpy as np
import os


def levenshtein_ratio_and_distance(s, t, ratio_calc=True):
    """Compute a distance between two strings

        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein distance between the first i characters of s and the
        first j characters of t.
        This function comes from : https://www.datacamp.com/community/tutorials/fuzzy-string-python
        Its goal is to define a distance between two strings, given potential charcater deletion and such.
        All comments in this function are from the original source

        :param s: (string) first string to be compared
        :param t: (string) second string to be compared
        :param ratio_calc:  (bool) toggles the type of distance used (number of edits or string distance)
        :return ratio: (float) Levenshtein distance betwwen the strings
    """

    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                     distance[row][col - 1] + 1,  # Cost of insertions
                                     distance[row - 1][col - 1] + cost)  # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])


def match_name_to_pkmn_name(detected_name, path_res):
    """ Match the detected name to the closest one existing in the name database, according to levenshtein distance

    :param detected_name: (string) detected name of the card by OCR
    :param path_res: (string) path to ressources, containing in particular the names of the 151 Pokemon
    :return: (string) closest existing name to the detected name
    """
    if len(detected_name) < 2:
        return "???????"

    # the name of all 151 Pokemon
    name_list = os.path.join(path_res, "151names.txt")
    print(path_res)

    # get all those names in a list
    names = []
    with open(name_list, encoding='utf-8-sig') as f:
        for line in f:
            names.append(line.rstrip())

    # go through all database names and measure their distance to the detected name.
    # return the one which has the lowest distance
    max_ratio = 0
    correct_name = ""
    for name in names:
        ratio = levenshtein_ratio_and_distance(detected_name, name)
        if ratio > max_ratio:
            correct_name = name
            max_ratio = ratio

    return correct_name

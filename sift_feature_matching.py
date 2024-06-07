# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Chargement des images gauche et droite
# img_left = cv2.imread('./img/input/2.jpg', cv2.IMREAD_GRAYSCALE)
# img_right = cv2.imread('./img/input/1.jpg', cv2.IMREAD_GRAYSCALE)

# # Initialisation du détecteur SIFT
# sift = cv2.SIFT_create()

# # Détectection des points clés et extraire des descripteurs pour les deux images
# keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
# keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

# # Utilisation de BFMatcher pour trouver les correspondances entre les descripteurs
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# matches = bf.match(descriptors_left, descriptors_right)

# # Tri des correspondances selon la distance
# matches = sorted(matches, key=lambda x: x.distance)

# # Dessin des correspondances
# img_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Affichage de l'image avec les correspondances
# plt.figure(figsize=(10, 10))
# plt.imshow(img_matches)
# plt.savefig('img/output/img_matches.jpg')  # Enregistrer la figure dans le dossier spécifié

# # Création d'une image en couleur pour afficher les correspondances
# img_left_color = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)

# # Traçage des lignes de correspondance sur l'image gauche
# for match in matches[:100]:  # Limiter à 100 correspondances pour plus de clarté
#     pt_left = tuple(map(int, keypoints_left[match.queryIdx].pt))
#     pt_right = tuple(map(int, keypoints_right[match.trainIdx].pt))
#     cv2.line(img_left_color, pt_left, pt_right, (0, 255, 0), 2)

# # Affichage de l'image gauche avec les lignes de correspondance
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB))
# plt.savefig('img/output/img_left_color.jpg')  # Enregistrement de la figure dans le dossier output


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Chargement des images en niveaux de gris
# def load_images(img_left_path, img_right_path):
#     return cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE), cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)

# # Détectection des points clés et calcule des descripteurs
# def detect_and_compute(img, detector):
#     return detector.detectAndCompute(img, None)

# def match_features(descriptors_left, descriptors_right):
#     # Utilisation de BFMatcher pour trouver les correspondances entre les descripteurs
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = bf.match(descriptors_left, descriptors_right)
#     # Tri des correspondances selon la distance
#     return sorted(matches, key=lambda x: x.distance)

# # Dessin des correspondances
# def draw_matches(img_left, keypoints_left, img_right, keypoints_right, matches):
#     return cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # Enregistrement de l'image output
# def save_image(img, path):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img)
#     plt.savefig(path)

# # Dessin des lignes de correspondance
# def draw_lines(img_left, keypoints_left, keypoints_right, matches):
#     img_left_color = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
#     for match in matches[:100]:
#         pt_left = tuple(map(int, keypoints_left[match.queryIdx].pt))
#         pt_right = tuple(map(int, keypoints_right[match.trainIdx].pt))
#         cv2.line(img_left_color, pt_left, pt_right, (0, 255, 0), 2)
#     return img_left_color

# def main():
#     img_left, img_right = load_images('./img/input/left.jpg', './img/input/rigth.jpg')
#     sift = cv2.SIFT_create()
#     keypoints_left, descriptors_left = detect_and_compute(img_left, sift)
#     keypoints_right, descriptors_right = detect_and_compute(img_right, sift)
#     matches = match_features(descriptors_left, descriptors_right)
#     img_matches = draw_matches(img_left, keypoints_left, img_right, keypoints_right, matches)
#     save_image(img_matches, 'img/output/img_matches.jpg')
#     img_left_color = draw_lines(img_left, keypoints_left, keypoints_right, matches)
#     save_image(cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB), 'img/output/img_left_color.jpg')

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np

def load_and_resize_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, size)

def detect_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2, ratio):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def filter_matches_by_distance(matches, max_distance=100):
    return [m for m in matches if m.distance < max_distance]

def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance=10):
    filtered_matches = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < max_vertical_distance:
            filtered_matches.append(match)
    return filtered_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

def main():
    size = (600, 800)
    image_gauche = load_and_resize_image('./img/input/left.jpg', size)
    image_droite = load_and_resize_image('./img/input/rigth.jpg', size)

    keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
    keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

    matches = match_features(descriptors_gauche, descriptors_droite, 0.6)  # Ajustement du ratio
    image_correspondances_initial = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
    cv2.imwrite('./img/output/Correspondances_Initiales_Sans_Filtre.jpg', image_correspondances_initial)

    # Appliquer les filtres un par un et observer les résultats
    matches = filter_matches_by_distance(matches, 200)  # Ajustement du seuil de distance
    image_correspondances_distance = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
    cv2.imwrite('./img/output/Correspondances_Après_Filtre_de_Distance.jpg', image_correspondances_distance)

    matches = filter_matches_by_horizontal_alignment(keypoints_gauche, keypoints_droite, matches, 30)  # Ajustement du seuil d'alignement horizontal
    image_correspondances_final = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
    cv2.imwrite('./img/output/Correspondances_Finale.jpg', image_correspondances_final)

if __name__ == "__main__":
    main()

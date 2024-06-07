import cv2
import numpy as np


def load_and_resize_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, size)

def detect_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def match_features(descriptors1, descriptors2, ratio):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def filter_matches_by_distance(matches, max_distance=100):
    # Filtrer les correspondances par distance
    return [m for m in matches if m.distance < max_distance]

def filter_matches_by_horizontal_alignment(keypoints1, keypoints2, matches, max_vertical_distance=10):
    # Filtrer les correspondances pour s'assurer qu'elles sont principalement horizontales
    filtered_matches = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < max_vertical_distance:
            filtered_matches.append(match)
    return filtered_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=2)

# def main():
#     size = (600, 800)
#     image_gauche = load_and_resize_image('./img/input/left.jpg', size)
#     image_droite = load_and_resize_image('./img/input/rigth.jpg', size)

#     keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
#     keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

#     matches = match_features(descriptors_gauche, descriptors_droite, 0.6)  # Ajustement du ratio
#     image_correspondances_initial = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imwrite('./img/output/Correspondances_Initiales_Sans_Filtre.jpg', image_correspondances_initial)

#     # Appliquer les filtres un par un et observer les résultats
#     matches = filter_matches_by_distance(matches, 200)  # Ajustement du seuil de distance
#     image_correspondances_distance = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imwrite('./img/output/Correspondances_Après_Filtre_de_Distance.jpg', image_correspondances_distance)

#     matches = filter_matches_by_horizontal_alignment(keypoints_gauche, keypoints_droite, matches, 30)  # Ajustement du seuil d'alignement horizontal
#     image_correspondances_final = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches)
#     cv2.imwrite('./img/output/Correspondances_Finale.jpg', image_correspondances_final)

# if __name__ == "__main__":
#     main()


def main():
    size = (600, 800)
    image_gauche = load_and_resize_image('./img/input/rigth.jpg', size)
    image_droite = load_and_resize_image('./img/input/left.jpg', size)

    keypoints_gauche, descriptors_gauche = detect_sift_features(image_gauche)
    keypoints_droite, descriptors_droite = detect_sift_features(image_droite)

    matches = match_features(descriptors_gauche, descriptors_droite, 0.9)  # Ajustement du ratio
    print(f"Nombre de correspondances initiales: {len(matches)}")
    # Convertir les points clés en une liste de coordonnées (x, y)
    points_gauche = np.float32([keypoints_gauche[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points_droite = np.float32([keypoints_droite[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Utiliser RANSAC pour trouver l'homographie
    _, mask = cv2.findHomography(points_gauche, points_droite, cv2.RANSAC, 2.0)

    # Convertir le masque en une liste de booléens
    mask = mask.ravel().tolist()

    # Filtrer les correspondances en utilisant le masque
    matches_ransac = [m for m, msk in zip(matches, mask) if msk]

    image_correspondances_ransac = draw_matches(image_gauche, keypoints_gauche, image_droite, keypoints_droite, matches_ransac)
    cv2.imwrite('./img/output/Correspondances_RANSAC.jpg', image_correspondances_ransac)

if __name__ == "__main__":
    main()
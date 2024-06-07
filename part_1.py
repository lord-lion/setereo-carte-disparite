import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger les images
img_left = cv2.imread('./img/input/left.jpg')
img_right = cv2.imread('./img/input/rigth.jpg')

# Convertir les images en niveaux de gris
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Initialiser le détecteur de points d'intérêt SIFT
sift = cv2.SIFT_create()

# Calculer les points d'intérêt SIFT et les descripteurs
keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)

# Utiliser le matcher BFMatcher pour faire la mise en correspondance
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

# Appliquer le ratio test de David Lowe pour sélectionner les bons matches
good_matches = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:  # Utiliser un seuil plus strict
        good_matches.append(m)

# Convertir les keypoints en coordonnées
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# Créer une image de sortie pour dessiner les correspondances
output_img = img_left.copy()

# Dessiner les lignes de correspondance sur l'image gauche
for pt_left, pt_right in zip(pts_left, pts_right):
    x1, y1 = pt_left
    x2, y2 = pt_right
    cv2.line(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    cv2.circle(output_img, (int(x1), int(y1)), 5, (0, 255, 255), -1)

# Afficher les résultats
# plt.figure(figsize=(10, 5))
# plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
# plt.title('Correspondances SIFT (Lignes jaunes sur l\'image gauche uniquement)')
# plt.show()
cv2.imwrite('./img/output/img_out.jpg', cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
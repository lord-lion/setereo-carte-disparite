import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(img_left_path, img_right_path):
    return cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE), cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)

def detect_and_compute(img, detector):
    return detector.detectAndCompute(img, None)

def match_features(descriptors_left, descriptors_right):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_left, descriptors_right)
    return sorted(matches, key=lambda x: x.distance)

def draw_matches(img_left, keypoints_left, img_right, keypoints_right, matches):
    return cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def apply_ransac(keypoints_left, keypoints_right, matches):
    pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    return [m for m, inlier in zip(matches, mask) if inlier]

def main():
    img_left, img_right = load_images('./img/input/left.jpg', './img/input/rigth.jpg')

    detectors = {
        'SIFT': cv2.SIFT_create(),
        'ORB': cv2.ORB_create(),
        'AKAZE': cv2.AKAZE_create()
    }

    for name, detector in detectors.items():
        keypoints_left, descriptors_left = detect_and_compute(img_left, detector)
        keypoints_right, descriptors_right = detect_and_compute(img_right, detector)
        matches = match_features(descriptors_left, descriptors_right)
        matches_ransac = apply_ransac(keypoints_left, keypoints_right, matches)
        
        img_matches = draw_matches(img_left, keypoints_left, img_right, keypoints_right, matches_ransac)
        plt.figure(figsize=(10, 10))
        plt.title(f'Keypoints Matching with {name}')
        plt.imshow(img_matches)
        plt.savefig(f'img/output/{name}_matches.png')  # Enregistrer l'image au lieu de la montrer

if __name__ == "__main__":
    main()

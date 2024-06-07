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

def apply_ransac(keypoints_left, keypoints_right, matches):
    pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches])
    pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches])

    # Utilisation de RANSAC pour trouver la transformation homogène
    H, mask = cv2.findHomography(pts_left, pts_right, cv2.RANSAC, 5.0)
    
    matches_mask = mask.ravel().tolist()
    return [m for i, m in enumerate(matches) if matches_mask[i]]

def draw_matches(img_left, keypoints_left, img_right, keypoints_right, matches):
    return cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def save_image(img, path):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.savefig(path)

def draw_lines(img_left, keypoints_left, keypoints_right, matches):
    img_left_color = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    for match in matches[:100]:
        pt_left = tuple(map(int, keypoints_left[match.queryIdx].pt))
        pt_right = tuple(map(int, keypoints_right[match.trainIdx].pt))
        cv2.line(img_left_color, pt_left, pt_right, (0, 255, 0), 2)
    return img_left_color

def main():
    img_left, img_right = load_images('./img/input/left.jpg', './img/input/rigth.jpg')
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = detect_and_compute(img_left, sift)
    keypoints_right, descriptors_right = detect_and_compute(img_right, sift)
    matches = match_features(descriptors_left, descriptors_right)
    matches_ransac = apply_ransac(keypoints_left, keypoints_right, matches)
    img_matches = draw_matches(img_left, keypoints_left, img_right, keypoints_right, matches_ransac)
    save_image(img_matches, 'img/output/img_matches_ransac.jpg')
    img_left_color = draw_lines(img_left, keypoints_left, keypoints_right, matches_ransac)
    save_image(cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB), 'img/output/img_left_color_ransac.jpg')

if __name__ == "__main__":
    main()

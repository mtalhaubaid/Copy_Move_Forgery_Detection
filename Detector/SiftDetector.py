import cv2

from Detector.AbstractDetector import AbstractDetector
from Detector.RansacDetector import RansacDetector
from DrawFunctions.Rectangle import DrawRectangle
# # copy-move forgery detection with sift
# class SiftDetector(AbstractDetector):
#     # blue
#     image = None
#     key_points = None
#     descriptors = None
#     color = (0, 0, 255)
#     distance = cv2.NORM_L2

#     def __init__(self, image):
#         self.image = image
#         self.detectFeature()
#         super().__init__(self.image)

#     # detect keypoints and descriptors
#     def detectFeature(self):
#         sift = cv2.SIFT_create()
#         gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
#         self.key_points, self.descriptors = sift.detectAndCompute(gray, None)
        
class SiftDetector(RansacDetector):
    color = (0, 0, 255)

    def detectFeature(self):
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)
        matches = self.match_features(self.descriptors, self.descriptors)
        self.detectForgery(matches)
        self.visualization_shape()

    def match_features(self, desc1, desc2, distance_ratio=0.7):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        good_matches = [m for m, n in matches if m.distance < distance_ratio * n.distance]
        return good_matches
    
    def visualization_shape(self):
        return DrawRectangle
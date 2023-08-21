import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from Detector.AbstractDetector import AbstractDetector
from DrawFunctions.Rectangle import DrawRectangle

class RansacDetector(AbstractDetector):
    def __init__(self, image):
        self.image = image.copy()
        self.detectFeature()
        self.matches = self.match_features(self.descriptors, self.descriptors)
        self.detectForgery(self.matches)
        super().__init__(self.image)

    def detectFeature(self):
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)

    def match_features(self, desc1, desc2, distance_ratio=0.7):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        good_matches = [m for m, n in matches if m.distance < distance_ratio * n.distance]
        return good_matches
    
    def detectForgery(self, matches, threshold=5):
        # Prepare data for clustering
        points = np.array([(self.key_points[match.queryIdx].pt, self.key_points[match.trainIdx].pt) for match in matches])
        points = points.reshape(points.shape[0], -1)

        # Clustering
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
        clustering.fit(points)

        for cluster_id in set(clustering.labels_):
            if list(clustering.labels_).count(cluster_id) > threshold:
                src_pts = np.float32([self.key_points[match.queryIdx].pt for idx, match in enumerate(matches) if clustering.labels_[idx] == cluster_id]).reshape(-1,1,2)
                dst_pts = np.float32([self.key_points[match.trainIdx].pt for idx, match in enumerate(matches) if clustering.labels_[idx] == cluster_id]).reshape(-1,1,2)
                
                # RANSAC to identify affine transformations
                if len(src_pts) >= 4 and len(dst_pts) >= 4:  # Ensure there are at least 4 points to compute homography
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask.sum() > threshold:
                        # Forgery detected for this cluster
                        # Visualization: Draw lines connecting the clustered keypoints
                        for i, (src, dst) in enumerate(zip(src_pts, dst_pts)):
                            if mask[i]:
                                pt1, pt2 = tuple(map(int, src[0])), tuple(map(int, dst[0]))
                                cv2.line(self.image, pt1, pt2, (0, 255, 0), 2)
            
    def visualization_shape(self):
        return DrawRectangle
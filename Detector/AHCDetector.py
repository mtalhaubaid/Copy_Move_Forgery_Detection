import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from Detector.AbstractDetector import AbstractDetector

from DrawFunctions.Line import DrawLine 

class AHCDetector(AbstractDetector):
    def __init__(self, image):
        self.image = image.copy()
        self.detectFeature()
        self.applyClustering()
        super().__init__(self.image)

    def detectFeature(self):
        # Using SIFT for demonstration purposes.
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)

    def applyClustering(self, threshold=5):
        # Prepare data for clustering
        points = np.array([kp.pt for kp in self.key_points])
        
        # Clustering
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
        clustering.fit(points)

        for cluster_id in set(clustering.labels_):
            if list(clustering.labels_).count(cluster_id) > threshold:
                cluster_points = [kp.pt for idx, kp in enumerate(self.key_points) if clustering.labels_[idx] == cluster_id]
                
                # Draw circles around each keypoint in the cluster
                for pt in cluster_points:
                    cv2.circle(self.image, tuple(map(int, pt)), 5, (0, 255, 0), -1)  # 5 is the radius of the circle
    
    def visualization_shape(self):
        return DrawLine
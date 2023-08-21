import cv2

from DrawFunctions.AbstractShape import AbstractShape

class DrawCircle(AbstractShape):
    image = None
    key_points = None
    color = None
    radius = 5  # Default radius for the circle

    def __init__(self, image, keypoints, color, radius=None):
        self.image = image
        self.key_points = keypoints
        self.color = color
        if radius:
            self.radius = radius
        self.draw()

    def DrawCircle(self, **kwargs):
        forgery = self.image.copy()
        for keypoint in self.key_points:
            cv2.circle(forgery, (int(keypoint[0]), int(keypoint[1])), self.radius, self.color, -1)  # -1 means fill the circle

        self.image = forgery
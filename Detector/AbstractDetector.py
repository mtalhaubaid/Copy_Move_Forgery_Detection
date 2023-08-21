from abc import ABCMeta, abstractmethod

from Detector.MatchFeature.Match import MatchFeatures
from DrawFunctions.Rectangle import DrawRectangle
from DrawFunctions.Line import DrawLine

# from DrawFunctions.Circle import DrawCircle
class AbstractDetector(metaclass=ABCMeta):
    key_points = None
    descriptors = None
    color = None
    image = None
    distance = None
    MatchFeatures = None
    Draw = None

    def __init__(self, image):
        self.image = image
        self.MatchFeatures = MatchFeatures(self.key_points, self.descriptors, self.distance)  # match points
        self.Draw = DrawRectangle(self.image, self.MatchFeatures.gPoint1, self.MatchFeatures.gPoint2, self.color, self.MatchFeatures.cRectangle)  # draw matches
        # self.Draw = DrawLine(self.image,  self.MatchFeatures.gPoint1,  self.MatchFeatures.gPoint2, self.color) # from DrawFunctions.Line import DrawLine -> import it
        # self.Draw = DrawCircle(self.image, self.MatchFeatures.gPoint1, self.MatchFeatures.gPoint2, self.color) # from DrawFunctions.Circle import DrawCircle -> import it
        self.image = self.Draw.image

    # detect keypoints and descriptors
    @abstractmethod
    def detectFeature(self):
        pass
    
    
    def visualize(self, key_points1, key_points2):
        shape_cls = self.visualization_shape()
        shape_cls(self.image, key_points1, key_points2, self.color)
        
    @abstractmethod
    def visualization_shape(self):
        raise NotImplementedError("Subclasses should implement this method to return the correct drawing class")
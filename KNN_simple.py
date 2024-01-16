import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

class K_Nearest_Neighbors():
    """
    This class is meant as an example to show how KNN algorithm works and takes a specific type of input
    an example of the type of data inputs it takes: 

    points = {'blue': [[2,4], [1,3], [2,3], [3,2], [2,1]],
          'orange': [[5,6], [4,5], [4,6], [6,6], [5,4]]}

    new_point = [3,3]

    The code of this class will have to be modified to handle pandas dataframes
    """
    def __init__(self, k=3):
        self.k = k
        self.point = None

    def fit(self, points):
        self.points = points

    def euclidean_distance(self, p, q):
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    def predict(self, new_point):
        distances = [] # we are going to be storing the all the distances here
        for category in self.points: # iterating through the different classes in the points dictionary
            for point in self.points[category]: # iterating through each point in each class
                distance = self.euclidean_distance(point, new_point) # computing the distance between each point and the new point
                distances.append([distance, category]) # appending the calculated distance along with the class it belongs to

        # the distances list contains the distances between each point and the new point and the class of each of those existing points
        categories = [category[1] for category in sorted(distances)[:self.k]] # now the categories list stores the classes of the k closest points
        result = mode(categories) # takes the mode of the k classes and returns that as the prediction
        return result
    

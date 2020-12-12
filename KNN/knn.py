import numpy as np
from collections import Counter


############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and label to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        self.feature_train = np.array(features)
        self.labels_train = np.array(labels)

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighbours.
        :param point: List[float]
        :return:  List[int]
        """
        dist = list(enumerate([(self.distance_function(self.feature_train[i], point), self.labels_train[i])
                       for i in range(len(self.feature_train))]))
        dist_sort = sorted(dist,key = lambda x:(x[1][0],x[0]))[:self.k]

        k_nearest_neighbours = [j[1][1] for j in dist_sort]
        return k_nearest_neighbours

    # TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to
        process every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each
        test data point, find the majority of labels for these neighbours as the predicted label for that testing
        data point (you can assume that k is always a odd number). Thus, you will get N predicted label for N test
        data point. This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        label_predict = []
        for i in features:
            neighbours = self.get_k_neighbors(i)
            label_count = Counter(neighbours)
            label_predict.append(max(label_count,key = lambda x:label_count[x]))
            # print(neighbours, label_count, label_predict)
        return label_predict

if __name__ == '__main__':
    print(np.__version__)

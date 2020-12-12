import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    assert len(real_labels) == len(predicted_labels)

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1 and real_labels[i] == 1:
            tp += 1
        elif predicted_labels[i] == 1 and real_labels[i] == 0:
            fp += 1
        elif real_labels[i] == 0 and predicted_labels[i] == 0:
            tn += 1
        else:
            fn += 1

    F1 = tp / (tp + (0.5 * (fp + fn)))

    return F1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        mink_dist = np.sum(np.absolute(np.subtract(p1, p2)) ** 3) ** (1 / 3)
        return mink_dist

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        euc_dist = np.sqrt(np.sum((p1 - p2) ** 2))
        return euc_dist

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        p1 = np.array(point1)
        p2 = np.array(point2)
        p1_norm = np.sqrt(np.sum(p1 ** 2))
        p2_norm = np.sqrt(np.sum(p2 ** 2))
        if p1_norm == 0 and p2_norm == 0:
            cosine_dist = 1
        else:
            cosine_dist = 1 - (np.dot(p1, p2) / (p1_norm * p2_norm))
        return cosine_dist


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values
        of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you
        need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        # print("input",distance_funcs)
        max_F1 = 0

        for distance_func in distance_funcs.items():
            for i in range(1, min(30, len(x_train) + 1), 2):

                knn_model = KNN(i, distance_func[1])
                knn_model.train(x_train, y_train)
                predict_val = knn_model.predict(x_val)
                # print(i, distance_func)
                current_F1 = f1_score(y_val, predict_val)
                if current_F1 > max_F1:
                    max_F1 = current_F1
                    self.best_k = i
                    self.best_distance_function = distance_func[0]
                    self.best_model = knn_model

        return

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers
        implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model,
        apply the scalers in scaling_classes to both of them.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you
        need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to
        try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign
        them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of
        scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

        max_F1 = 0
        for scaler in scaling_classes.items():
            # print(scaler)
            scaler_obj = scaler[1]()
            scaled_train_features = scaler_obj(x_train)
            scaled_val_features = scaler_obj(x_val)
            for distance_func in distance_funcs.items():
                for i in range(1, min(30, len(x_train) + 1), 2):

                    knn_model = KNN(i, distance_func[1])
                    knn_model.train(scaled_train_features, y_train)
                    predict_val = knn_model.predict(scaled_val_features)
                    # print(i, distance_func)
                    current_F1 = f1_score(y_val, predict_val)
                    # print("F1", current_F1, "distance_func",distance_func[0], scaler[0])
                    if current_F1 > max_F1:
                        # print("best",i, distance_func, current_F1)
                        max_F1 = current_F1
                        self.best_k = i
                        self.best_distance_function = distance_func[0]
                        self.best_model = knn_model
                        self.best_scaler = scaler[0]


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features_np = np.array(features)
        features_norm = [np.sqrt(np.sum(i)) for i in (features_np ** 2)]
        normalised = []
        for j in range(len(features)):
            if features_norm[j] != 0:
                normalised.append([k / features_norm[j] for k in features[j]])
            else:
                normalised.append(features[j])
        return normalised


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features_np = np.array(features)
        feature_max_min = []
        for i in range(len(features_np[0])):
            feature_max_min.append([max(features_np[:, i]), min(features_np[:, i])])
        result = []
        for j in range(len(features)):
            curr = []
            for k in range(len(features[j])):
                if feature_max_min[k][0] != feature_max_min[k][1]:
                    curr.append(
                        (features[j][k] - feature_max_min[k][1]) / (feature_max_min[k][0] - feature_max_min[k][1]))
                else:
                    curr.append(0)
            result.append(curr)

        return result

import numpy as np


#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    ###############################################
    # TODO: implement the Kmeans++ initialization
    ###############################################
    centers = []
    k1 = generator.randint(0, n)
    centers.append(k1)
    while len(centers) < n_cluster:
        distance = []
        centers_data = x[centers]
        for i in range(x.shape[0]):
            distance.append(np.min(np.square(np.linalg.norm((centers_data - x[i]), axis=1))))
        normalised_distance = distance / np.sum(distance)
        p = generator.rand()
        cumulative = np.cumsum(normalised_distance).tolist()
        for m in range(n):
            if cumulative[m] >= p:
                centers.append(m)
                break
    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################

        def assignment(centroids):
            distance_centers = np.ones((N, self.n_cluster))
            for i in range(self.n_cluster):
                distance_centers[:, i] = np.square(np.linalg.norm((x - centroids[i]), axis=1))
            gamma = np.argmin(distance_centers, axis=1)
            return gamma

        def distortion(gamma, centroids):
            distortion_value = np.sum(np.square(np.linalg.norm(x - centroids[gamma], axis=1)))
            return distortion_value

        def new_centroids(gamma):
            centroids_new = np.array([np.mean(x[gamma == j], axis=0) for j in range(self.n_cluster)])
            return centroids_new

        centroids = x[self.centers]
        new_gamma = assignment(centroids)
        distort_val_old = distortion(new_gamma, centroids)
        new_centers = new_centroids(new_gamma)
        update_no = 1
        for iter in range(self.max_iter - 1):
            new_gamma = assignment(new_centers)
            new_distort_val = distortion(new_gamma, new_centers)
            update_no += 1
            if abs(new_distort_val - distort_val_old) < self.e:
                break
            else:
                distort_val_old = new_distort_val
                new_centers = new_centroids(new_gamma)
        print(update_no)
        return new_centers, new_gamma, update_no


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented,
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        kmeans_obj = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, gamma, iterations = kmeans_obj.fit(x, centroid_func)
        centroid_labels = []
        for i in range(self.n_cluster):
            unique, counts = np.unique(y[gamma == i], return_counts=True)
            counter = dict(zip(unique, counts))
            centroid_labels.append(max(counter, key=counter.get))
        centroid_labels = np.array(centroid_labels)

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        distances = []
        for i in range(self.n_cluster):
            centroid_NxD = np.tile(self.centroids[i], (N, 1))
            distances.append(np.square(np.linalg.norm((x - centroid_NxD), axis=1)))
        distances = np.array(distances)
        centroid_index = np.argmin(distances.reshape(self.n_cluster, N).T, axis=1)
        predicted = self.centroid_labels[centroid_index]
        return predicted


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    n1, n2, D = image.shape
    data = image.reshape(n1 * n2, D)
    distances = []
    for i in range(len(code_vectors)):
        centroid_NxD = np.tile(code_vectors[i], (n1 * n2, 1))
        distances.append(np.square(np.linalg.norm((data - centroid_NxD), axis=1)))
    distances = np.array(distances)
    centroid_index = np.argmin(distances.reshape(len(code_vectors), n1 * n2).T, axis=1)
    updated_image = code_vectors[centroid_index].reshape(n1, n2, D)
    return updated_image

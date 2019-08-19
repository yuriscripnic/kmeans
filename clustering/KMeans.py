"""
KMeans clustering

This class classify a array of points using the k-means method

Parameters:
  n_cluster : The number of cluster to be used in the classification
  tol: the tolerance diffrence between the current set of centroids and the the previous
  n_inter: maximum number of iterations

  Methods:
  fit(X): Classify an array of points in the format [x,y] by the number of cluster k
  predict(A): Given an array of points , predict the cluster that it belongs

  Variables:
  labels : List representing the cluster id for each point
  cluster_centers: List of the coordinates of the center of each cluster

"""

import math
import random
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


class KMeans:

    def __init__(self, n_cluster=8, tol=1e-4, n_inter=10):
        self.n_cluster = n_cluster
        self.tol = tol
        self.n_inter = n_inter
        self.labels=[]
        self.cluster_centers=[]

    def euclidean_dist(self, a, b):
        (x1, y1) = a
        (x2, y2) = b
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def euclidean_dist_list(self, C, C_old):
        res = [self.euclidean_dist(C[i], C_old[i]) for i in range(self.n_cluster)]
        return sum(res)

    def mean_dist(self, elements):
        x = [e[0] for e in elements]
        y = [e[1] for e in elements]
        if(x and y):
            return [sum(x) / len(x), sum(y) / len(y)]
        else:
            return [math.inf,math.inf]

    def random_cluster_centers(self, X):
        max_x = max([x[0] for x in X])
        max_y = max([x[1] for x in X])
        return [[random.randrange(max_x),random.randrange(max_y)] for i in range(self.n_cluster)]

    def fit(self, X):
        n = len(X)
        self.cluster_centers = self.random_cluster_centers(X)
        logger.debug(f"random cluster centers: {self.cluster_centers}")
        self.labels = [None for i in range(len(X))]
        inertia = math.inf
        inter = 0
        while inertia > self.tol and inter < self.n_inter:
            elements = [[] for i in range(self.n_cluster)]
            old_centers = list(self.cluster_centers)
            for i in range(n):
                min_dist = math.inf
                for c in range(self.n_cluster):
                    dist = self.euclidean_dist(self.cluster_centers[c], X[i])
                    if dist < min_dist:
                        min_dist = dist
                        self.labels[i] = c
                elements[self.labels[i]].append(X[i])
            for c in range(self.n_cluster):
                self.cluster_centers[c] = self.mean_dist(elements[c])
            inertia = self.euclidean_dist_list(self.cluster_centers, old_centers)
            inter+=1

            logger.debug(f"Interation # {inter}/{self.n_inter}")
            logger.debug(f"cluster centers: {self.cluster_centers}")
            logger.debug(f"labels: {self.labels}")
            logger.debug(f"elements: {elements}")
            logger.debug(f"inertia: {inertia}")
        return self.cluster_centers

    def predict(self, X):
        n = len(X)
        res = [[] for i in range(n)]
        for i in range(n):
            min_dist = math.inf
            for c in range(self.n_cluster):
                dist = self.euclidean_dist(self.cluster_centers[c], X[i])
                if dist < min_dist:
                    min_dist = dist
                    res[i] = c
        return res

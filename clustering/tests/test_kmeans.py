import pytest
from clustering import KMeans

kmeans = KMeans(n_cluster=3)


def test_euclidean_dist():
    assert kmeans.euclidean_dist([0,0],[3,4]) == 5.0

def test_euclidean_dist_list():
    C = [[0,0],[0,1],[0,2]]
    C_old = [[0,1],[0,2],[0,3]]
    assert kmeans.euclidean_dist_list(C,C_old) == 3

def test_random_cluster_centers():
    res = kmeans.random_cluster_centers([[0,0],[1,1]])
    assert len(res) == 3
    assert max([x[0] for x in res]) < 1
    assert max([x[1] for x in res]) < 1

def test_mean_dist():
    assert kmeans.mean_dist([[0,0],[2,2]]) == [1,1]

def test_fit():
    res = kmeans.fit([[0,0],[10,10],[20,20]])
    assert len(res) == 3

## This test can fail if the result couldn't converge to a glogal minimum
def test_predict():
    kmeans.fit([[0, 0], [10, 10], [20, 20]])
    assert kmeans.predict([[1,1]]) == [0]



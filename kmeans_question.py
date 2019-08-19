"""
KMean Question:

Given a set of two dimensional points P (e.g. [(1.1, 2.5), (3.4,1.9)...]; the size of set can be
100s), write a function that calculates simple K-means. The expected returned value from the
function is 1) a set of cluster id that each point belongs to, and 2) coordinates of centroids at the
end of iteration.
Although you can write this in any language, we would recommend for you to use python.
Please feel free to research and look up any information you need, but please note plagiarism
will not be tolerated

This app calculate K-means using simple random values to initialize the centroids
Because this, the algorithm is not sure to find a global minimum and could diverge
Consecutive tries can result in a correct response

"""

from clustering import KMeans
import logging
import random
from matplotlib import pyplot as plt


plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

logging.getLogger("clustering.KMeans").setLevel(logging.DEBUG)

def main():
    k = 3
    X =   [[random.randint(0,20),random.randint(0,20)] for i in range(30)]       \
        + [[random.randint(40,60), random.randint(40,60)] for i in range(30)]    \
        + [[random.randint(80, 100), random.randint(80, 100)] for i in range(30)]

    print(f"Cluster points:{X}")

    kmeans = KMeans(n_cluster=k, tol=3e-4)
    centroids = kmeans.fit(X)
    prediction = kmeans.predict([[0.0,0.0],[50.0,40.0],[100.0,100.0]])

    print(f"KMeans centroids: {centroids}")
    print(f"KMeans predict for [0,0],[50,40],[100,100]]: {prediction}")

    colors = ['r', 'g', 'b']
    for i in range(k):
            plt.scatter([x[0] for x in X], [x[1] for x in X], s=7, c=colors[i])
    plt.scatter([x[0] for x in centroids], [x[1] for x in centroids], marker='*', s=200, c='black')
    plt.show()

if __name__ == "__main__":
    main()

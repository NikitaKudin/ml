#!/usr/local/Cellar/python/2.7.6/bin/python
# -*- coding: utf-8 -*-

'''Standard python modules'''
import sys

'''For scientific computing'''
from numpy import *
import scipy.misc, scipy.io, scipy.optimize, scipy.cluster.vq

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import findClosestCentroids as fcc
import runkMeans as rkm
import kMeansInitCentroids as kmic


'''For plotting'''
from matplotlib import pyplot, cm, colors


from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

def findClosestCentroids(X, centroids):
    K = shape(centroids)[0]
    m = shape(X)[0]
    idx = zeros((m, 1))

    for i in range(0, m):
        lowest = 999
        lowest_index = 0

        for k in range(0, K):
            cost = X[i] - centroids[k]
            cost = cost.T.dot(cost)
            if cost < lowest:
                lowest_index = k
                lowest = cost

        idx[i] = lowest_index
    return idx + 1  # add 1, since python's index starts at 0

def computeCentroidsLoop(X, idx, K):
    m, n = shape(X)
    centroids = zeros((K, n))

    for k in range(1, K + 1):

        counter = 0
        cum_sum = 0
        for i in range(0, m):
            if idx[i] == k:
                cum_sum += X[i]
                counter += 1
        centroids[k - 1] = cum_sum / counter
    return centroids

def computeCentroids(X, idx, K):
    m, n = shape(X)
    centroids = zeros((K, n))

    data = c_[X, idx]  # append the cluster index to the X

    for k in range(1, K + 1):
        temp = data[data[:, n] == k]  # quickly extract X that falls into the cluster
        count = shape(temp)[0]  # count number of entries for that cluster

        for j in range(0, n):
            centroids[k - 1, j] = sum(temp[:, j]) / count

    return centroids

def runkMeans(X, initial_centroids, max_iters, plot=False):
    K = shape(initial_centroids)[0]
    centroids = copy(initial_centroids)
    idx = None

    for iteration in range(0, max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)

        if plot is True:
            data = c_[X, idx]

            # Extract data that falls in to cluster 1, 2, and 3 respectively, and plot them out
            data_1 = data[data[:, 2] == 1]
            pyplot.plot(data_1[:, 0], data_1[:, 1], 'ro', markersize=5)

            data_2 = data[data[:, 2] == 2]
            pyplot.plot(data_2[:, 0], data_2[:, 1], 'go', markersize=5)

            data_3 = data[data[:, 2] == 3]
            pyplot.plot(data_3[:, 0], data_3[:, 1], 'bo', markersize=5)

            pyplot.plot(centroids[:, 0], centroids[:, 1], 'k*', markersize=17)

            pyplot.show(block=True)

    return centroids, idx

# 2.Реализуйте функцию случайной инициализации K центров кластеров.
def kMeansInitCentroids(X, K):
    return random.permutation(X)[:K]


def part_1():
    # 1.Загрузите данные ex6data1.mat из файла.
    print('Part 1')
    mat = scipy.io.loadmat("ex6data1.mat")
    X = mat['X']
    K = 3

    initial_centroids = array([[3, 3], [6, 2], [8, 5]])

    idx = findClosestCentroids(X, initial_centroids)
    print(idx[0:3])  # should be [1, 3, 2]

    centroids = computeCentroids(X, idx, K)
    print(centroids)


# should be
# [[ 2.428301  3.157924]
#  [ 5.813503  2.633656]
#  [ 7.119387  3.616684]]


def part_2():

    print('Part 2')
    mat = scipy.io.loadmat("ex6data1.mat")
    X = mat['X']
    #K = 3

    max_iters = 10
    centroids = array([[3, 3], [6, 2], [8, 5]])

    # 5.Реализуйте алгоритм K-средних.
    # 6.Постройте график, на котором данные разделены на K=3 кластеров (при помощи различных маркеров или цветов), а также траекторию движения центров кластеров в процессе работы алгоритма
    runkMeans(X, centroids, max_iters, plot=True)


def part_3():
    print('Part 3')
    mat = scipy.io.loadmat("ex6data1.mat")
    X = mat['X']
    K = 3

    max_iters = 10
    centroids = array([[3, 3], [6, 2], [8, 5]])
    print(kMeansInitCentroids(X, K))  # it's randomly one of the coordinates from X


def part_4_1():
    ## ============= Part 4: K-Means Clustering on Pixels ===============
    #  In this exercise, you will use K-Means to compress an image. To do this,
    #  you will first run K-Means on the colors of the pixels in the image and
    #  then you will map each pixel on to it's closest centroid.
    #
    #  You should now complete the code in kMeansInitCentroids.m
    #

    print('\nRunning K-Means clustering on pixels from an image.\n\n')

    # 7.Загрузите данные bird_small.mat из файла.
    #  Load an image of a bird
    mat = scipy.io.loadmat('bird_small.mat')
    A = mat["A"]

    A = A / 255.0  # Divide by 255 so that all values are in the range 0 - 1

    # Size of the image
    img_size = A.shape

    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = A.reshape(img_size[0] * img_size[1], 3, order='F').copy()

    # Run your K-Means algorithm on this data
    # You should try different values of K and max_iters here
    K = 16
    max_iters = 10

    # When using K-Means, it is important the initialize the centroids
    # randomly.
    # You should complete the code in kMeansInitCentroids.m before proceeding
    initial_centroids = kmic.kMeansInitCentroids(X, K)

    # Run K-Means
    centroids, idx = rkm.runkMeans(X, initial_centroids, max_iters)

    ## ================= Part 5: Image Compression ======================
    #  In this part of the exercise, you will use the clusters of K-Means to
    #  compress an image. To do this, we first find the closest clusters for
    #  each example. After that, we

    print('\nApplying K-Means to compress an image.\n')

    # Find closest cluster members
    idx = fcc.findClosestCentroids(X, centroids)

    # Essentially, now we have represented the image X as in terms of the
    # indices in idx.

    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by it's index in idx) to the centroid value
    X_recovered = centroids[idx, :]

    # Reshape the recovered image into proper dimensions
    X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3, order='F')

    # Display the original image
    plt.close()
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title('Original')

    # Display compressed image side by side
    plt.subplot(1, 2, 2)
    plt.imshow(X_recovered)
    plt.title('Compressed, with {:d} colors.'.format(K))
    plt.show(block=False)
    plt.savefig('results1.png')
    print('results1.png has bee saved')


def part_4_2():
    img = io.imread('airplane.png')
    img_r = (img / 255.0).reshape(-1, 3)
    # Fit K-means on resized image. n_clusters is the desired number of colors
    k_colors = KMeans(n_clusters=16).fit(img_r)
    # Assign colors to pixels based on their cluster center
    # Each row in k_colors.cluster_centers_ represents the RGB value of a cluster centroid
    # k_colors.labels_ contains the cluster that a pixel is assigned to
    # The following assigns every pixel the color of the centroid it is assigned to
    img128 = k_colors.cluster_centers_[k_colors.labels_]
    # Reshape the image back to 128x128x3 to save
    img128 = np.reshape(img128, (img.shape))
    # Save image
    image.imsave('results2.png', img128)
    print('results2.png has bee saved')


def main():
    set_printoptions(precision=6, linewidth=200)
    part_1()
    part_2()
    part_3()
    part_4_1()

    # 10.	Реализуйте алгоритм K-средних на другом изображении.
    part_4_2()

    # TODO: 11.Реализуйте алгоритм иерархической кластеризации на том же изображении. Сравните полученные результаты.

if __name__ == '__main__':
    main()

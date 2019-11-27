#Набор данных ex9_movies.mat представляет собой файл формата *.mat (т.е. сохраненного из Matlab). 
#Набор содержит две матрицы Y и R - рейтинг 1682 фильмов среди 943 пользователей. 
#Значение Rij может быть равно 0 или 1 в зависимости от того оценил ли пользователь j фильм i. 
#Матрица Y содержит числа от 1 до 5 - оценки в баллах пользователей, выставленные фильмам.
#Задание.
#DONE: 1.	Загрузите данные ex9_movies.mat из файла.
#DONE: 2.	Выберите число признаков фильмов (n) для реализации алгоритма коллаборативной фильтрации.
# подставнока X, Theta ??? 3.	Реализуйте функцию стоимости для алгоритма.
#DONE: 4.	Реализуйте функцию вычисления градиентов.
#DONE: 5.	При реализации используйте векторизацию для ускорения процесса обучения.
#DONE: 6.	Добавьте L2-регуляризацию в модель.
#DONE: 7.	Обучите модель с помощью градиентного спуска или других методов оптимизации.
#DONE: 8.	Добавьте несколько оценок фильмов от себя. Файл movie_ids.txt содержит индексы каждого из фильмов.
#DONE: 9.	Сделайте рекомендации для себя. Совпали ли они с реальностью?
#10.	Также обучите модель с помощью сингулярного разложения матриц. Отличаются ли полученные результаты?
#11.	Ответы на вопросы представьте в виде отчета.


#!/usr/bin/env python

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cofiCostFunc as ccf
import checkCostFunction as chcf
import loadMovieList as lml
import normalizeRatings as nr
import SVD as svd

## =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.
#  
print('Loading movie ratings dataset.\n')

#  Load data
mat = scipy.io.loadmat('ex9_movies.mat')
Y = mat["Y"]
R = mat["R"]

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
#print('Average rating for movie 1 (Toy Story): {:f} / 5\n'.format(np.mean(Y[0, R[0, :]==1])))

print('Average movies rating: {:f}'.format(Y[1,R[1,:]].mean()));

#  We can "visualize" the ratings matrix by plotting it with imagesc
# need aspect='auto' for a ~16:9 (vs almost vertical) aspect
plt.imshow(Y, aspect='auto');
plt.ylabel('Movies');
plt.xlabel('Users');
#plt.show()

## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
#mat = scipy.io.loadmat('.mat')
#X = mat["X"]
#Theta = mat["Theta"]
#num_users = mat["num_users"]
#num_movies = mat["num_movies"]
#num_features = mat["num_features"]

#  Reduce the data set size so that this runs faster
num_users = 4 
num_movies = 5 
num_features = 3

X =  np.random.rand(num_movies, num_features);
Theta =  np.random.rand(num_users, num_features);

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

#  Evaluate cost function
params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))
J, _ = ccf.cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)

print('Cost at loaded parameters: {:f}\n'.format(J))

#('Program paused. Press enter to continue.')


## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement 
#  the collaborative filtering gradient function. Specifically, you should 
#  complete the code in cofiCostFunc.m to return the grad argument.
#  
print('\nChecking Gradients (without regularization) ... \n')

#  Check gradients by running checkNNGradients
chcf.checkCostFunction()

## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for 
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#  

#  Evaluate cost function
params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))
J, _ = ccf.cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
           
print('Cost at loaded parameters (lambda_var = 1.5): {:f} '\
         '\n(this value should be about 31.34)\n'.format(J))


## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement 
#  regularization for the gradient. 
#

#  
print('\nChecking Gradients (with regularization) ... \n')

#  Check gradients by running checkNNGradients
chcf.checkCostFunction(1.5)

## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#
movieList = lml.loadMovieList()

#  Initialize my ratings
my_ratings = np.zeros((1682, 1))

# NOTE THAT THE FOLLOWING SECTION AS WELL AS THE movie_ids.txt file
# USED HERE IS ADAPTED FOR PYTHON'S 0-INDEX (VS MATLAB/OCTAVE'S 1-INDEX)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 0, so to rate it "4", you can set
# my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
# my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[26] = 5
my_ratings[2] = 5
#my_ratings[11]= 5
## my_ratings[53] = 4
#my_ratings[63]= 4
## my_ratings[65]= 3
#my_ratings[68] = 3
#my_ratings[97] = 4
## my_ratings[182] = 4
#my_ratings[225] = 3
# my_ratings[354]= 5

print('\n\nNew user ratings:\n')
for i, rating in enumerate(my_ratings):
    if rating > 0: 
        print('Rated {:.0f} for {:s}\n'.format(rating[0], movieList[i]))


## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating 
#  dataset of 1682 movies and 943 users
#

print('\nTraining collaborative filtering...\n')

#  Load data
mat = scipy.io.loadmat('ex9_movies.mat')
Y = mat["Y"]
R = mat["R"]

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.column_stack((my_ratings, Y))
R = np.column_stack(((my_ratings != 0).astype(int), R))

#  Normalize Ratings
[Ynorm, Ymean] = nr.normalizeRatings(Y, R)

#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

# Set options
maxiter = 100
options = {'disp': True, 'maxiter':maxiter}
lambda_var=10

# Create "short hand" for the cost function to be minimized
def costFunc(initial_parameters):
    return ccf.cofiCostFunc(initial_parameters, Y, R, num_users, num_movies, num_features, lambda_var)

# Set Regularization
results = minimize(costFunc, x0=initial_parameters, options=options, method="L-BFGS-B", jac=True)
theta = results["x"]

# Unfold the returned theta back into U and W
X = np.reshape(theta[:num_movies*num_features], (num_movies, num_features), order='F')
Theta = np.reshape(theta[num_movies*num_features:], (num_users, num_features), order='F')

print('Recommender system learning completed.\n')

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#

p = np.dot(X, Theta.T)
my_predictions = p[:,0] + Ymean.flatten()

movieList = lml.loadMovieList()

# from http://stackoverflow.com/a/16486305/583834
# reverse sorting by index
ix = my_predictions.argsort()[::-1]

print('\n\nTop Grad recommendations for you:\n')
for i in range(9):
    j = ix[i]
    print('Predicting rating {:.5f} for movie {:s}'.format(my_predictions[j]-2, movieList[j]))

print('\n\nOriginal ratings provided:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated {:d} for {:s}'.format(int(my_ratings[i]), movieList[i]))

## ================== Part 8: Matrix SVD ====================

#res = svd.SVD(mat = A, initial_mat1 = B, initial_mat2 = C, learn_rate = alpha, iterations = N)

#from sklearn.datasets import make_regression
#X, y, coefficients = make_regression(
#    n_samples=50,
#    n_features=1,
#    n_informative=1,
#    n_targets=1,
#    noise=5,
#    coef=True,
#    random_state=1
#)

#X = np.random.randn(num_movies, num_features)
#Theta = np.random.randn(num_users, num_features)
#y = Y;


##initial_parameters = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

#n = X.shape[1]
#r = np.linalg.matrix_rank(X)

#U, sigma, VT = np.linalg.svd(X, full_matrices=False)

#D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))

#V = VT.T

#X_plus = V.dot(D_plus).dot(U.T)

#svd_results = X_plus.dot(y);

#aa = svd_results;

# ----------------------------- 2

#import scipy.sparse as sp
#from scipy.sparse.linalg import svds

#user_item_ratings = Y;

#U, S, VT = svds(user_item_ratings, k = 20)
#S_diagonal = np.diag(S)
#Y_hat = np.dot(np.dot(U, S_diagonal), VT)

#aa = Y_hat;


# -------------------------------3 ---------------

##from scipy.sparse import csc_matrix
#from scipy.sparse.linalg import svds
#A = np.random.randn(num_movies, num_features);

#k = min(A.shape);

##csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
#u, s, vt = svds(A, k=9) # k is the number of factors
#s

#==

#k = min(Y.shape) - 1;

##csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)

#initial_parameters = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

#u, s, vt = svds(Y, k) # k is the number of factors

#theta = s;

## Unfold the returned theta back into U and W
#X = np.reshape(theta[:num_movies*num_features], (num_movies, num_features), order='F')
#Theta = np.reshape(theta[num_movies*num_features:], (num_users, num_features), order='F')

#print('Recommender system learning completed.\n')

### ================== Part 8: Recommendation for you ====================
##  After training the model, you can now make recommendations by computing
##  the predictions matrix.
##

#p = np.dot(X, Theta.T)
#my_predictions = p[:,0] + Ymean.flatten()

#movieList = lml.loadMovieList()

## from http://stackoverflow.com/a/16486305/583834
## reverse sorting by index
#ix = my_predictions.argsort()[::-1]

#print('\n\nTop recommendations for you:\n')
#for i in range(10):
#    j = ix[i]
#    print('Predicting rating {:.5f} for movie {:s}'.format(my_predictions[j], movieList[j]))

#print('\n\nOriginal ratings provided:')
#for i in range(len(my_ratings)):
#    if my_ratings[i] > 0:
#        print('Rated {:d} for {:s}'.format(int(my_ratings[i]), movieList[i]))

#------------------------------- 4 ---------------------------------

from sklearn.decomposition import TruncatedSVD

X = Y.T
X.shape
#Fitting the Model
SVD = TruncatedSVD(n_components=num_features, random_state=0)
matrix = SVD.fit_transform(X)
matrix.shape

corr = np.corrcoef(matrix)
corr.shape

#title = Y.columns
#title_list = list(title)
#samia = title_list.index('Memoirs of a Geisha')
#corr_samia  = corr[samia]
#list(title[(corr_samia >= 0.9)])

#p = np.dot(X, Theta.T)
#my_predictions = p[:,0] + Ymean.flatten()

shtitle = Y # columns

movieList = lml.loadMovieList()
bad_boys_idx = 26;
four_rooms_idx = 2;

corr_bad_boys  = corr[bad_boys_idx]
corr_four_rooms  = corr[four_rooms_idx]

corr_bad_boys_sorted = np.sort(corr_bad_boys)[::-1]
corr_four_rooms_sorted = np.sort(corr_four_rooms)[::-1]

#list(title[(corr_bad_boys >= 0.9)])

recommended_films = np.where(corr_bad_boys_sorted >= 0.98)[0]
recommended_films2 = np.where(corr_four_rooms_sorted >= 0.98)[0]

#recommended_films = recommended_films.argsort()[::1]

#ix = my_predictions.argsort()[::-1]

print('\n\nTop SVD recommendations for you:\n')
for i in range(1,10):
    val = corr_bad_boys_sorted[i]
    index = np.where(corr_bad_boys == val)[0][0]
    print('Predicting rating {:.5f} for movie {:s}'.format(val*5, movieList[index]))

recommended_films


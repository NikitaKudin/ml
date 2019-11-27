# please, make sure to import numpy
import numpy as np

# the function SVD takes parameters such as:
# learning rate
alpha = 0.001
# number of iterations
N = 1000
# the m by n matrix that we want to approximate
A = np.array([[3, 2], [-4, 6], [-1, 3]])
# and a couple of matrices: B is m by k
B = np.array([[1, 2], [4, 1], [2, 2]])
# and C is n by k
C = np.array([[-3, 1], [2, 3]])

# run returns updated B, C, and approximated A that is a multiplications of the B and C
SVD(mat = A, initial_mat1 = B, initial_mat2 = C, learn_rate = alpha, iterations = N)

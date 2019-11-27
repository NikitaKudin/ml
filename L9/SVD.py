def SVD(mat, initial_mat1, initial_mat2, learn_rate, iterations):
    ## reassigning values in order to keep code clean
    # the m by n matrix that we want to approximate
    A = mat
    # two matrices from which we will start: B is m by k
    B = initial_mat1
    # C is n by k
    C = initial_mat2
    # learning rate, or step of learning
    alpha = learn_rate
    # number of iterations
    N = iterations
    # A ~ B * C^t : the first approximation based on given initial matrices
    A_app = np.dot(B, C.T)
    # gradient descent
    for i in range(N):
        # partial derivatives for matrices
        dLdB = np.dot((A_app - A), C)
        dLdC = np.dot((A_app - A).T, B)
        # updating matrices
        C = C - alpha * dLdC
        B = B - alpha * dLdB
        # calculating approximated matrix
        A_app = np.dot(B, C.T)
    # returning two matrices that can be used for A approximation
    return B, C, A_app

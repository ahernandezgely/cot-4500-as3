import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)

def function(t: float, w: float):
    return t - (w**2)

def Eulers():
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    h = (end_of_t - start_of_t) / num_of_iterations
    w = original_w
    t = start_of_t
    
    for cur_iteration in range(0, num_of_iterations):
        w = w + h * function(t, w)
        t = t + h

    print("%.5f" % w, '\n')

def RungeKutta():
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    h = (end_of_t - start_of_t) / num_of_iterations
    w = original_w
    t = start_of_t
    
    for cur_iteration in range(0, num_of_iterations):
        k1 = h * function(t, w)
        k2 = h * function(t + h/2, w + k1/2)
        k3 = h * function(t + h/2, w + k2/2)
        k4 = h * function(t + h, w + k3)
        
        w = w + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        t = t + h

    print("%.5f" % w, '\n')

def GaussElim(A, b):
    n = len(b)
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)
    
    for i in range(0, n - 1):
        t = i + 1
        while Ab[i][i] == 0:
                if t < n:
                    Ab[[i, t]] = Ab[[t, i]]
                    t += 1
                else:
                    print("no solution \n")
                    return
        for j in range(i + 1,n):
            if Ab[j][i] == 0:
                continue
            m = Ab[j][i]/Ab[i][i]
            Ab[j][:] = Ab[j][:] - m * Ab[i][:]
    
    x = np.zeros_like(b, dtype=np.double)
    x[n - 1] = Ab[n - 1][n] / Ab[n -1][n -1]
    
    for i in range(n-2, -1, -1):
        sum = 0
        for j in range(i+1,n):
            sum += Ab[i][j] * x[j]
        x[i] = (Ab[i][n] - sum) / Ab[i][i]

    print(x, '\n')

def luFactorization(A):
    n = len(A)
    U = A
    L = np.eye(n)
    
    for i in range(0, n - 1):
        t = i + 1
        while U[i][i] == 0:
                if t < n:
                    U[[i, t]] = U[[t, i]]
                    t += 1
                else:
                    print("no solution \n")
                    return
        for j in range(i + 1,n):
            if U[j][i] == 0:
                L[j][i] = 0
                continue
            m = U[j][i]/U[i][i]
            L[j][i] = m
            U[j][:] = U[j][:] - m * U[i][:]
    
    determinant = 1.0
    for i in range(0, n):
        determinant = determinant * U[i][i]
        
    print("%.5f" % determinant, '\n')
    print(L, '\n')
    print(U, '\n')

def diagDominate(A):
    n = len(A)
    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            if not (i == j):
                sum += A[i][j]
        if A[i][i] < sum:
            return False
    
    return True     

def posDefinitive(A):
    if np.array_equal(A, A.T):
        eigen = np.linalg.eigvals(A)
        n = len(eigen)
        
        for i in range(0, n):
            if not (eigen[i] > 0):
                return False

        return True
    else:
        return False

if __name__ == "__main__":

    Eulers()

    RungeKutta()

    A = np.array([[2,-1,1],
                  [1,3,1],
                  [-1,5,4]])

    b = np.array([6,0,-3])

    GaussElim(A, b)

    A = np.array([[1, 1, 0, 3],
                  [2, 1, -1, 1],
                  [3, -1, -1, 2],
                  [-1, 2, 3, -1]], dtype='f')

    luFactorization(A)

    A = np.array([[9, 0, 5, 2, 1],
                  [3, 9, 1, 2, 1],
                  [0, 1, 7, 2, 3],
                  [4, 2, 3, 12, 2],
                  [3, 2, 4, 0, 8]])

    print(diagDominate(A), '\n')

    A = np.array([[2, 2, 1],
                  [2, 3, 0],
                  [1, 0, 2]])

    print(posDefinitive(A), '\n')
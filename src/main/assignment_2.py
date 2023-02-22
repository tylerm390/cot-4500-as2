import numpy
numpy.set_printoptions(precision=7,suppress=True,linewidth=100)
# PROBLEM !: NEVILLES ITERATED INTERPOLATION
def nevilles(x, xs, ys):
    n = len(xs)
    Q = [[0] * (n) for j in range(n)]
    for i in range(n):
        Q[i][0] = ys[i]

    for i in range(1, n):
        for j in range(1, i+1):
            Q[i][j] = ((x - xs[i-j]) * Q[i][j-1] - (x - xs[i]) * Q[i-1][j-1]) / (xs[i] - xs[i-j])

    return Q

Q = nevilles(3.7, [3.6, 3.8, 3.9], [1.675, 1.436, 1.318])
print(Q[2][2])

# PROBLEM 2 AND 3: NEWTON'S FORWARD DIFFERENCES
def newtons(x, xs, ys):
    n = len(xs)
    F = [[0] * n for j in range(n)]
    for i in range(n):
        F[i][0] = ys[i]

    for i in range(1, n):
        for j in range(1, i+1):
                F[i][j] = (F[i][j-1] - F[i-1][j-1]) / (xs[i] - xs[i-j])
    
    F_diag = [F[i][i] for i in range(1, n)]
    fx = 0
    for i in range(n):
        cur = F[i][i]
        for j in range(i):
            cur *= x - xs[j]
        fx += cur
    return (F_diag, fx)

F, fx = newtons(7.3, [7.2, 7.4, 7.5, 7.6], [23.5492, 25.3913, 26.8224, 27.4589])
print(F)
print(fx)

# PROBLEM 4: HERMITE INTERPOLATION
def hermite(xs, ys, dys):
    n = len(xs)
    Q = numpy.zeros((2*n, 2*n));
    z = numpy.zeros(2*n);

    for i in range(n):
        z[2*i] = xs[i]
        z[2*i+1] = xs[i]
        Q[2*i][0] = ys[i]
        Q[2*i+1][0] = ys[i]
        Q[2*i+1][1] = dys[i]

        if not i == 0:
            Q[2*i][1] = (Q[2*i][0] - Q[2*i-1][0]) / (z[2*i] - z[2*i-1])

    for i in range(2, 2*n):
        for j in range(2, i+1):
            Q[i][j] = (Q[i][j-1] - Q[i-1][j-1]) / (z[i] - z[i-j])
    
    out = numpy.zeros((2*n, 2*n+1))

    for i in range (2*n):
        out[i][0] = z[i]
        for j in range(1, 2*n + 1):
            out[i][j] = Q[i][j-1]

    return out

Q = hermite([3.6, 3.8, 3.9], [1.675, 1.436, 1.318], [-1.195, -1.188, -1.182])

print(Q)

# PROBLEM 5: CUBIC SPLINES

def splines(xs, ys):
    n = len(xs)
    h = [xs[i+1] - xs[i] for i in range(n-1)]
    A = numpy.zeros((n, n))

    A[0][0] = A[n-1][n-1] = 1
    for i in range(1, n-1):
        A[i][i-1] = h[i-1]
        A[i][i] = 2 * (h[i-1] + h[i])
        A[i][i+1] = h[i]

    b = numpy.zeros(n)
    for i in range(1, n-1):
        b[i] = 3 * (ys[i+1] - ys[i]) / h[i] - 3 * (ys[i] - ys[i-1]) / h[i-1]

    l = [0] * n
    m = [0] * n
    z = [0] * n
    
    l[0] = 1

    for i in range(1, n-1):
        l[i] = 2*(xs[i+1] - xs[i-1]) - h[i-1] * m[i-1]
        m[i] = h[i] / l[i]
        z[i] = (b[i] - h[i-1]*z[i-1]) / l[i]

    l[n-1] = 1
    
    x = numpy.zeros(n)
    for j in range(n-2, -1, -1):
        x[j] = z[j] - m[j]*x[j+1]

    return (A, b, x)

A, b, x = splines([2, 5, 8, 10], [3, 5, 7, 9])
print(A)
print(b)
print(x)








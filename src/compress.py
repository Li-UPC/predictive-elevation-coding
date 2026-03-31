import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.linalg import solve


def lagrange12(oldz):
    """
    Computes the 12 optimal Lagrange predictor coefficients for a 2D integer matrix.

    For each interior element z[i][j], the predictor estimates its value as a linear
    combination of 12 neighboring elements (5 from the current/previous row, 7 from
    the two rows above). The coefficients are found by solving a 13x13 linear system
    that minimizes the sum of squared prediction errors over the entire matrix.

    The matrix is zero-padded on the right (4 extra columns) so that neighbors of
    elements near the right edge are always defined.

    Returns a 1D array of 12 floats (the predictor coefficients).
    """
    z = np.hstack((oldz, np.zeros((oldz.shape[0], 4), dtype=int)))

    n = oldz.shape[0]
    m = oldz.shape[1]

    z0 = z[2:n, 2:m]
    z1 = z[2:n, 1:m-1]
    z2 = z[1:n-1, 1:m-1]
    z3 = z[1:n-1, 2:m]
    z4 = z[1:n-1, 3:m+1]
    z5 = z[1:n-1, 4:m+2]

    b1 = z[2:n, 0:m-2]
    b2 = z[1:n-1, 0:m-2]
    b3 = z[0:n-2, 0:m-2]
    b4 = z[0:n-2, 1:m-1]
    b5 = z[0:n-2, 2:m]
    b6 = z[0:n-2, 3:m+1]
    b7 = z[0:n-2, 4:m+2]

    matrices = {
        'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7
    }

    index  = ['z1', 'z2', 'z3', 'z4', 'z5', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
    indexz = ['z1', 'z2', 'z3', 'z4', 'z5']
    indexb = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
    gram = pd.DataFrame(0.0, index=index, columns=index)

    # Build symmetric Gram matrix of dot products between neighbor submatrices
    for i in range(len(index)):
        for j in range(len(index)):
            f = index[i]
            c = index[j]
            if i <= j:
                gram.loc[f, c] = np.sum(matrices[f] * matrices[c])
            else:
                gram.loc[f, c] = gram.loc[c, f]

    # Build the 13x13 linear system (12 predictor coefficients + 1 bias term)
    A = np.zeros((13, 13))
    b = np.zeros(13)

    f = 0
    for z in indexz:
        c = 0
        for i in index:
            A[f][c] = gram.loc[z, i]
            c += 1
        f += 1

    for ib in indexb:
        c = 0
        for i in index:
            A[f][c] = gram.loc[ib, i]
            c += 1
        f += 1

    i = 0
    for e in index:
        c = np.sum(matrices[e])
        A[i][-1] = -0.5 * c
        A[-1][i] = c
        b[i] = np.sum(matrices[e] * z0)
        i += 1

    b[-1] = np.sum(z0)

    sol = solve(A, b)

    return sol


def main():
    if len(sys.argv) != 4:
        print("Usage: python compress.py <c|d> <infile> <outfile>")
        sys.exit(1)

    flag = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3]

    if flag == 'c':
        z = np.loadtxt(infile, dtype=int)
        # Compute the 12 Lagrange predictor coefficients from the input matrix.
        # These are passed to the C++ executable, which uses them during compression
        # and stores them in the file header for later decompression.
        sol = lagrange12(z)
        subprocess.run(["./compressor", "c", infile, outfile,
                        str(sol[0]), str(sol[1]), str(sol[2]), str(sol[3]),
                        str(sol[4]), str(sol[5]), str(sol[6]), str(sol[7]),
                        str(sol[8]), str(sol[9]), str(sol[10]), str(sol[11])])
    elif flag == 'd':
        subprocess.run(["./compressor", "d", infile, outfile])


if __name__ == "__main__":
    main()

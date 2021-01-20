#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import degrees as deg
import numpy as np
from optparse import OptionParser
import pandas as pd
from sympy import cos
from sympy import lambdify
from sympy import Matrix
from sympy import sin
from sympy import symbols


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def calculateH(xa, ya, XA, YA, ZA, f):
    """Compute height of exposure station with formula 6-13."""
    xai = xa[:-1]
    xbi = xa[1:]
    yai = ya[:-1]
    ybi = ya[1:]
    XAi = XA[:-1]
    XBi = XA[1:]
    YAi = YA[:-1]
    YBi = YA[1:]
    ZAi = ZA[:-1]
    ZBi = ZA[1:]

    AB = (np.sqrt((XAi - XBi)**2 + (YAi - YBi)**2))

    a = ((xbi-xai)/f)**2 + ((ybi-yai)/f)**2
    b = (2*((xbi-xai)/f)*((xai/f)*ZAi - (xbi/f)*ZBi)) + \
        (2*((ybi-yai)/f)*((yai/f)*ZAi - (ybi/f)*ZBi))
    c = ((xai/f)*ZAi-(xbi/f)*ZBi)**2 + ((yai/f)*ZAi - (ybi/f)*ZBi)**2 - AB**2

    # Find the two roots of H
    sol = np.dstack((
                    (-b + np.sqrt(b**2 - 4*a*c))/(2*a),
                    (-b - np.sqrt(b**2 - 4*a*c))/(2*a))).reshape(-1, 2)
    H = sol.max(axis=1)

    H = H[~np.isnan(H)]
    return H[abs(H - H.mean()) < H.std()].mean()


def conformalTrans(xa, ya, XA, YA):
    """Perform conformal transformation parameters."""
    # Define coefficient matrix
    n = xa.shape[0]     # Number of control points
    B = np.matrix(np.zeros((2 * n, 4)))
    B[:n, 2] = B[n:, 3] = 1
    B[:n, 0] = B[n:, 1] = xa
    B[:n, 1] = -ya
    B[n:, 0] = ya

    # Define constants matrix
    f = np.matrix(np.concatenate((XA, YA)))

    # Compute transformation parameters
    a, b, Tx, Ty = np.array(((B.T*B).I*(B.T)*f)).ravel()

    return a, b, Tx, Ty


def getInit(xa, ya, XA, YA, ZA, f):
    """Compute initial values of unknown parameters."""
    Omega = Phi = 0
    H = calculateH(xa, ya, XA, YA, ZA, f)   # Attitude of exposure station

    # Compute arbitrary horizontal coordinates with formula 6-5, 6-6
    XA2 = xa * (H-ZA) / f
    YA2 = ya * (H-ZA) / f

    # Perform conformal transformation
    a, b, XL, YL = conformalTrans(XA2, YA2, XA, YA)

    Kappa = np.arctan2(b, a)

    return XL, YL, H, Omega, Phi, Kappa


def getM(Omega, Phi, Kappa):
    """Compute rotation matrix M."""
    M = np.matrix([
        [
            cos(Phi)*cos(Kappa),
            sin(Omega)*sin(Phi)*cos(Kappa) + cos(Omega)*sin(Kappa),
            -cos(Omega)*sin(Phi)*cos(Kappa) + sin(Omega)*sin(Kappa)],
        [
            -cos(Phi)*sin(Kappa),
            -sin(Omega)*sin(Phi)*sin(Kappa) + cos(Omega)*cos(Kappa),
            cos(Omega)*sin(Phi)*sin(Kappa) + sin(Omega)*cos(Kappa)],
        [
            sin(Phi),
            -sin(Omega)*cos(Phi),
            cos(Omega)*cos(Phi)]
        ])

    return M


def getEqn(IO, EO, PT, pt):
    """List observation equations."""
    f, xo, yo = IO
    XL, YL, ZL, Omega, Phi, Kappa = EO
    XA, YA, ZA = PT
    xa, ya = pt

    M = getM(Omega, Phi, Kappa)

    r = M[0, 0] * (XA - XL) + M[0, 1] * (YA - YL) + M[0, 2] * (ZA - ZL)
    s = M[1, 0] * (XA - XL) + M[1, 1] * (YA - YL) + M[1, 2] * (ZA - ZL)
    q = M[2, 0] * (XA - XL) + M[2, 1] * (YA - YL) + M[2, 2] * (ZA - ZL)

    F = Matrix([xa - xo + f * (r / q), ya - yo + f * (s / q)])
    return F


def estimate(sample, f, s, funcObj, init):
    """Compute the model parameters with sample point sets."""
    # Define input observables
    xa, ya, XA, YA, ZA, SigX, SigY, SigZ = np.hsplit(sample.values, 8)

    # Define weight matrix
    err = np.dstack((SigX, SigY, SigZ)).reshape(1, -1)  # Error vector
    W = np.matrix(np.diag(s**2 / (err**2).ravel()))
    Q = W.I

    numPt = len(xa)

    # Compute initial values if the initial file is not specified
    if not init:
        X0 = np.matrix(getInit(xa, ya, XA, YA, ZA, f)).T
    else:
        X0 = np.matrix(pd.read_csv(
            init,
            delimiter=' ',
            usecols=range(6),
            names=[str(i) for i in range(6)]).values).T

    print("Initial Values:\n Param\tValue")
    print("Omega\t%.6f\tdeg." % X0[3, 0])
    print("Phi\t%.6f\tdeg." % X0[4, 0])
    print("Kappa\t%.6f\tdeg." % X0[5, 0])
    print("XL\t%.6f" % X0[0, 0])
    print("XL\t%.6f" % X0[1, 0])
    print("ZL\t%.6f" % X0[2, 0])
    print()

    dX = np.ones(1)                              # Initial value for iteration

    # Create array for the observables and initial values
    l = np.zeros((numPt, 11))
    l[:, :6] += X0[:, :].T
    l[:, 6] += XA.ravel()
    l[:, 7] += YA.ravel()
    l[:, 8] += ZA.ravel()
    l[:, 9] += xa.ravel()
    l[:, 10] += ya.ravel()

    # Iteration process
    lc = 0          # Loop count
    dRes = 1.       # Termination criteria
    res = 1.        # Initial value of residual
    FuncJFl, FuncJFx, FuncF = funcObj
    while dRes > 10**-12 and lc < 20:
        # Compute coefficient matrix and constants matrix
        A = np.zeros((2 * numPt, err.shape[1]))
        B = np.zeros((2 * numPt, 6))

        Ai = FuncJFl(*np.hsplit(l, 11)[:-2])
        Bi = FuncJFx(*np.hsplit(l, 11)[:-2])
        F0 = np.matrix(-FuncF(*np.hsplit(l, 11)).T.reshape(-1, 1))

        for i in range(numPt):
            A[2*i:2*(i+1), 3*i:3*(i+1)] = Ai[:, :, i].reshape(2, 3)
            B[2*i:2*(i+1), :] = Bi[:, :, i].reshape(2, 6)

        A = np.matrix(A)
        B = np.matrix(B)

        AT = A.T.copy()
        Qe = (A * Q * AT)
        We = Qe.I
        N = (B.T * We * B)                  # Compute normal matrix
        t = (B.T * We * F0)                 # Compute t matrix
        dX = N.I * t                        # Compute unknown parameters
        V = Q * AT * We * (F0 - B * dX)     # Compute residual vector

        X0 = X0 + dX            # Update initial values
        l[:, :6] += dX[:, :].T

        # Update termination criteria
        if lc > 1:
            dRes = abs(((V.T * W * V)[0, 0]/res) - 1)
        res = (V.T * W * V)[0, 0]

        # Compute sigma0
        s0 = (res / (B.shape[0] - B.shape[1]))**0.5

        lc += 1

    return X0, s0, N


def getInlier(data, f, s, funcObj, X, thres):
    """Get the index of inlier."""
    # Define input observables
    xa, ya, XA, YA, ZA, SigX, SigY, SigZ = np.hsplit(data.values, 8)

    # Define weight matrix
    err = np.dstack((SigX, SigY, SigZ)).reshape(1, -1)  # Error vector
    W = np.matrix(np.diag(s**2 / (err**2).ravel()))
    Q = W.I

    numPt = len(data)

    # Create observable array as argument of function objects
    l = np.zeros((numPt, 11))
    l[:, :6] += X[:, :].T
    l[:, 6] += XA.ravel()
    l[:, 7] += YA.ravel()
    l[:, 8] += ZA.ravel()
    l[:, 9] += xa.ravel()
    l[:, 10] += ya.ravel()

    # Compute coefficient matrix and constants matrix
    A = np.zeros((2 * numPt, err.shape[1]))
    B = np.zeros((2 * numPt, 6))

    FuncJFl, FuncJFx, FuncF = funcObj
    Ai = FuncJFl(*np.hsplit(l, 11)[:-2])
    Bi = FuncJFx(*np.hsplit(l, 11)[:-2])
    F0 = np.matrix(-FuncF(*np.hsplit(l, 11)).T.reshape(-1, 1))
    for i in range(numPt):
        A[2*i:2*(i+1), 3*i:3*(i+1)] = Ai[:, :, i].reshape(2, 3)
        B[2*i:2*(i+1), :] = Bi[:, :, i].reshape(2, 6)

    A = np.matrix(A)
    B = np.matrix(B)

    AT = A.T.copy()
    Qe = (A * Q * AT)
    We = Qe.I
    N = (B.T * We * B)
    t = (B.T * We * F0)
    dX = N.I * t
    V = Q * AT * We * (F0 - B * dX)

    # Get the inlier mask
    dis = np.sqrt(np.power(V.reshape(-1, 3), 2).sum(axis=1))
    mask = (dis < thres)

    return data[mask].index


def spaceResection(inputFile, outputFile, s,
                   useRANSAC, maxIter, sampleSize, thres, init):
    """Perform a space resection."""
    # Read observables from txt file
    with open(inputFile) as fin:
        f = float(fin.readline())           # The focal length in mm

    # Define symbols
    EO = symbols("XL YL ZL Omega Phi Kappa")  # Exterior orienration parameters
    PT = symbols("XA YA ZA")    # Object point coordinates
    pt = symbols("xa ya")       # Image coordinates

    # Define variable for inerior orienration parameters
    IO = f, 0, 0

    # List and linearize observation equations
    F = getEqn(IO, EO, PT, pt)
    JFx = F.jacobian(EO)
    JFl = F.jacobian(PT)    # Jacobian matrix for observables

    # Create lambda function objects
    FuncJFl = lambdify((EO+PT), JFl, 'numpy')
    FuncJFx = lambdify((EO+PT), JFx, 'numpy')
    FuncF = lambdify((EO+PT+pt), F, 'numpy')

    data = pd.read_csv(
        inputFile,
        delimiter=' ',
        usecols=range(1, 9),
        names=[str(i) for i in range(8)],
        skiprows=1)

    # Check data size
    if useRANSAC and len(data) <= sampleSize:
        print("Insufficient data for applying RANSAC method,")
        print("change to normal approach")
        useRANSAC = False

    if useRANSAC:
        bestErr = np.inf
        bestIC = 0
        bestParam = 0
        bestN = 0
        for i in range(maxIter):
            print("Iteration count: %d" % (i+1))
            sample = data.sample(sampleSize)
            # Compute initial model with sample data
            try:
                X0, s0, N = estimate(
                    sample, f, s, (FuncJFl, FuncJFx, FuncF), init)
            except np.linalg.linalg.LinAlgError:
                continue

            idx = getInlier(data, f, s, (FuncJFl, FuncJFx, FuncF), X0, thres)
            consensusSet = data.loc[idx]    # Inliers

            # Update the model if the number consesus set is greater than
            # current model and the error is smaller
            if len(consensusSet) >= bestIC:
                try:
                    X0, s0, N = estimate(
                        consensusSet, f, s, (FuncJFl, FuncJFx, FuncF), init)
                except np.linalg.linalg.LinAlgError:
                    continue

                if s0 < bestErr:
                    bestErr = s0
                    bestIC = len(consensusSet)
                    bestParam = X0
                    bestN = N
                    print("Found better model,")
                    print("inlier=%d (%.2f%%), error=%.6f" % \
                        (bestIC, 100.0 * bestIC / len(data), bestErr))

        if bestIC == 0:
            print("Cannot apply RANSAC method, change to normal approach")
            bestParam, bestErr, bestN = estimate(
                data, f, s, (FuncJFl, FuncJFx, FuncF), init)
    else:
        bestParam, bestErr, bestN = estimate(
            data.sample(frac=1), f, s, (FuncJFl, FuncJFx, FuncF), init)

    # Compute other informations
    SigmaXX = bestErr**2 * bestN.I
    paramStd = np.sqrt(np.diag(SigmaXX))
    XL, YL, ZL, Omega, Phi, Kappa = np.array(bestParam).ravel()

    # Output results
    print("Exterior orientation parameters:")
    print((" %9s %11s %11s") % ("Parameter", "Value", "Std."))
    print(" %-10s %11.6f %11.6f" % (
        "Omega(deg)", deg(Omega) % 360, deg(paramStd[3])))
    print(" %-10s %11.6f %11.6f" % (
        "Phi(deg)", deg(Phi) % 360, deg(paramStd[4])))
    print(" %-10s %11.6f %11.6f" % (
        "Kappa(deg)", deg(Kappa) % 360, deg(paramStd[5])))
    print(" %-10s %11.6f %11.6f" % ("XL", XL, paramStd[0]))
    print(" %-10s %11.6f %11.6f" % ("YL", YL, paramStd[1]))
    print(" %-10s %11.6f %11.6f" % ("ZL", ZL, paramStd[2]))
    print("\nSigma0 : %.6f" % bestErr)

    with open(outputFile, 'w') as fout:
        fout.write("%.6f "*3 % (XL, YL, ZL))
        fout.write("%.6f "*3 %
                   tuple(map(lambda x: deg(x) % 360, [Omega, Phi, Kappa])))
        fout.write("%.6f "*3 % tuple(paramStd[:3]))
        fout.write("%.6f "*3 % tuple(map(lambda x: deg(x), paramStd[3:])))


def main():
    parser = OptionParser(usage="%prog [options]", version="%prog 0.2")

    # Define options
    parser.add_option(
        '-i', '--input',
        default="input.txt",
        help="read input data from FILE, default value is \"input.txt\"",
        metavar='FILE')

    parser.add_option(
        '-o', '--output',
        default="result.txt",
        help="name of output file, default value is \"result.txt\"",
        metavar='FILE')

    parser.add_option(
        '-s', '--sigma',
        type='float',
        dest='s',
        default=0.005,
        help="define a priori error, default value is 0.005",
        metavar='N')

    parser.add_option(
        '-R', '--RANSAC',
        action='store_true',
        dest='R',
        default=False,
        help="use RANSAC method, default value is False")

    parser.add_option(
        '-m', '--max',
        type='int',
        dest='m',
        default=5,
        help="maximum number of iterations of RANSAC, default value is 5",
        metavar='N')

    parser.add_option(
        '-n', '--num',
        type='int',
        dest='n',
        default=10,
        help="sample size for initinal model of RANSAC, default value is 10",
        metavar='N')

    parser.add_option(
        '-t', '--threshold',
        type='float',
        dest='t',
        default=0.01,
        help="threshold for RANSAC, default value is 0.01",
        metavar='N')

    parser.add_option(
        '-I', '--init',
        default=None,
        help="use initial value from the specified E.O. file",
        metavar='FILE')

    # Instruct optparse object
    (options, args) = parser.parse_args()

    spaceResection(options.input, options.output, options.s,
                   options.R, options.m, options.n, options.t, options.init)

    return 0


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import sin, cos, Matrix, symbols, lambdify
from optparse import OptionParser
from math import degrees as deg
import numpy as np
import pandas as pd


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
    a, b, Tx, Ty = np.array(((B.T*B).I*(B.T)*f)).flatten()

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


def spaceResection(inputFile, outputFile, s):
    """Perform a space resection"""
    # Define symbols
    EO = symbols("XL YL ZL Omega Phi Kappa")  # Exterior orienration parameters
    PT = symbols("XA YA ZA")    # Object point coordinates
    pt = symbols("xa ya")       # Image coordinates

    # Read observables from txt file
    with open(inputFile) as fin:
        f = float(fin.readline())           # The focal length in mm

    data = pd.read_csv(
        inputFile,
        delimiter=' ',
        usecols=range(1, 9),
        names=[str(i) for i in range(8)],
        skiprows=1)

    xa, ya, XA, YA, ZA, SigX, SigY, SigZ = np.hsplit(data.values, 8)

    # Compute initial values
    X0 = np.matrix(getInit(xa, ya, XA, YA, ZA, f)).T

    print "Initial Values:\n Param\tValue"
    print "  Omega\t%.6f\tdeg." % deg(X0[3, 0])
    print "  Phi\t%.6f\tdeg." % deg(X0[4, 0])
    print "  Kappa\t%.6f\tdeg." % deg(X0[5, 0])
    print "   XL\t%.6f" % X0[0, 0]
    print "   YL\t%.6f" % X0[1, 0]
    print "   ZL\t%.6f" % X0[2, 0]
    print

    # Define variable for inerior orienration parameters
    IO = f, 0, 0

    # List and linearize observation equations
    F = getEqn(IO, EO, PT, pt)
    JFx = F.jacobian(EO)
    JFl = F.jacobian(PT)    # Jacobian matrix for observables

    # Create lambda function objects
    FuncJFx = lambdify((EO+PT), JFx, 'numpy')
    FuncJFl = lambdify((EO+PT), JFl, 'numpy')
    FuncF = lambdify((EO+PT+pt), F, 'numpy')

    # Define weight matrix
    err = np.dstack((SigX, SigY, SigZ)).reshape(1, -1)  # Error vector
    W = np.matrix(np.diag(s**2 / (err**2).ravel()))

    dX = np.ones(1)                              # Initial value for iteration

    numPt = len(xa)

    # Array for the observables and initial values
    # (XL, YL, ZL, Omega, Phi, Kappa, XA, YA, ZA, xa, ya)
    l = np.zeros((numPt, 11))
    l[:, 0] += X0[0, 0]
    l[:, 1] += X0[1, 0]
    l[:, 2] += X0[2, 0]
    l[:, 3] += X0[3, 0]
    l[:, 4] += X0[4, 0]
    l[:, 5] += X0[5, 0]
    l[:, 6] += XA.ravel()
    l[:, 7] += YA.ravel()
    l[:, 8] += ZA.ravel()
    l[:, 9] += xa.ravel()
    l[:, 10] += ya.ravel()

    # Iteration process
    lc = 0
    dRes = 1.       # Termination criteria
    res = 1.        # Initial value of residual
    while dRes > 10**-12 and lc < 20:
        # Compute coefficient matrix and constants matrix
        A = np.zeros((2 * numPt, err.shape[1]))
        B = np.zeros((2 * numPt, 6))
        f = np.zeros((2 * numPt, 1))

        Ai = FuncJFl(*np.hsplit(l, 11)[:-2])
        Bi = FuncJFx(*np.hsplit(l, 11)[:-2])
        f = np.matrix(-FuncF(*np.hsplit(l, 11)).T.reshape(-1, 1))
        for i in range(numPt):
            A[2*i:2*(i+1), 3*i:3*(i+1)] = Ai[:, :, i].reshape(2, 3)
            B[2*i:2*(i+1), :] = Bi[:, :, i].reshape(2, 6)

        B = np.matrix(B)

        Qe = (A * W.I * A.T)
        We = Qe.I
        N = (B.T * We * B)                  # Compute normal matrix
        t = (B.T * We * f)                  # Compute t matrix
        dX = N.I * t                        # Compute unknown parameters
        V = W.I * A.T * We * (f - B * dX)   # Compute residual vector

        X0 += dX            # Update initial values
        l[:, 0] += dX[0, 0]
        l[:, 1] += dX[1, 0]
        l[:, 2] += dX[2, 0]
        l[:, 3] += dX[3, 0]
        l[:, 4] += dX[4, 0]
        l[:, 5] += dX[5, 0]

        # Update termination criteria
        if lc > 1:
            dRes = abs(((V.T * W * V)[0, 0]/res) - 1)
        res = (V.T * W * V)[0, 0]

        # Compute sigma0
        s0 = (res / (B.shape[0] - B.shape[1]))**0.5

        lc += 1

    # Compute other informations
    SigmaXX = s0**2 * N.I
    paramStd = np.sqrt(np.diag(SigmaXX))
    XL, YL, ZL, Omega, Phi, Kappa = np.array(X0).ravel()

    # Output results
    print "Exterior orientation parameters:"
    print (" %9s %11s %11s") % ("Parameter", "Value", "Std.")
    print " %-10s %11.6f %11.6f" % (
        "Omega(deg)", deg(Omega) % 360, deg(paramStd[3]))
    print " %-10s %11.6f %11.6f" % (
        "Phi(deg)", deg(Phi) % 360, deg(paramStd[4]))
    print " %-10s %11.6f %11.6f" % (
        "Kappa(deg)", deg(Kappa) % 360, deg(paramStd[5]))
    print " %-10s %11.6f %11.6f" % ("XL", XL, paramStd[0])
    print " %-10s %11.6f %11.6f" % ("YL", YL, paramStd[1])
    print " %-10s %11.6f %11.6f" % ("ZL", ZL, paramStd[2])
    print "\nSigma0 : %.6f" % s0

    with open(outputFile, 'w') as fout:
        fout.write("%.6f "*3 % (XL, YL, ZL))
        fout.write("%.6f "*3 %
            tuple(map(lambda x: deg(x) % 360, [Omega, Phi, Kappa])))
        fout.write("%.6f "*3 % tuple(paramStd[:3]))
        fout.write("%.6f "*3 % tuple(map(lambda x: deg(x), paramStd[3:])))


def main():
    parser = OptionParser(usage="%prog [-i] [-o] [-s]", version="%prog 0.2")

    # Define options
    parser.add_option(
        '-i', '--input',
        help="read input data from FILE, the default value is \"input.txt\"",
        metavar='FILE')

    parser.add_option(
        '-o', '--output',
        help="name of output file, the default value is \"result.txt\"",
        metavar='OUTPUT')

    parser.add_option(
        '-s', '--sigma',
        type='float',
        dest='s',
        help="define a priori error, the default value is 0.005",
        metavar='N')

    # Instruct optparse object
    (options, args) = parser.parse_args()

    # Define default values if there are nothing given by the user
    if not options.input:
        options.input = "input.txt"

    if not options.output:
        options.output = "result.txt"

    if not options.s:
        options.s = 0.005

    spaceResection(options.input, options.output, options.s)

    return 0


if __name__ == '__main__':
    main()

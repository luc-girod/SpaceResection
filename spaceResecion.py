#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sympy import sin, cos, Matrix, symbols, lambdify
from optparse import OptionParser
from math import degrees as deg
from math import radians as rad
import numpy as np


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def calculateH(xa, ya, XA, YA, ZA, f):
    """Compute height of exposure station with formula 6-13"""
    xa, xb = xa[:2]
    ya, yb = ya[:2]
    XA, XB = XA[:2]
    YA, YB = YA[:2]
    ZA, ZB = ZA[:2]

    AB = (np.sqrt((XA - XB)**2 + (YA - YB)**2))

    a = ((xb-xa)/f)**2 + ((yb-ya)/f)**2
    b = (2*((xb-xa)/f)*((xa/f)*ZA - (xb/f)*ZB)) + \
        (2*((yb-ya)/f)*((ya/f)*ZA - (yb/f)*ZB))
    c = ((xa/f)*ZA-(xb/f)*ZB)**2 + ((ya/f)*ZA - (yb/f)*ZB)**2 - AB**2

    if (-b + np.sqrt(b**2 - 4*a*c))/(2*a) > 0:
        return ((-b + np.sqrt(b**2 - 4*a*c))/(2*a))[0]
    else:
        return ((-b - np.sqrt(b**2 - 4*a*c))/(2*a))[0]


def conformalTrans(xa, ya, XA, YA):
    """Perform conformal transformation parameters"""

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
    """Compute initial values of unknown parameters"""
    Omega = Phi = 0
    H = calculateH(xa, ya, XA, YA, ZA, f)   # Attitude of exposure station

    # Compute arbitrary horizontal coordinates with formula 6-5, 6-6
    XA2 = xa*(H-ZA)/f
    YA2 = ya*(H-ZA)/f

    # Perform conformal transformation
    a, b, XL, YL = conformalTrans(XA2, YA2, XA, YA)

    Kappa = np.arctan2(b, a)

    return XL, YL, H, Omega, Phi, Kappa


def getM(Omega, Phi, Kappa):
    """Compute rotation matrix M"""
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
    """List observation equations"""
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


def spaceResection(inputFile="input.txt", s=rad(5)):
    """Perform a space resection"""
    # Define symbols
    EO = symbols("XL YL ZL Omega Phi Kappa")  # Exterior orienration parameters
    PT = symbols("XA YA ZA")    # Object point coordinates
    pt = symbols("xa ya")       # Image coordinates

    # Read observables from txt file
    fin = open(inputFile)
    lines = fin.readlines()
    fin.close()

    f = float(lines[0])     # The focal length in mm
    Name, xa, ya, XA, YA, ZA, SigX, SigY, SigZ = np.hsplit(
        np.array(map(lambda x: x.split(), lines[1:])), 9)

    xa, ya, XA, YA, ZA, SigX, SigY, SigZ = map(
        lambda x: x.astype(np.double), [xa, ya, XA, YA, ZA, SigX, SigY, SigZ])

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
    FuncJFx = lambdify((EO+PT), JFx)
    FuncJFl = lambdify((EO+PT), JFl)
    FuncF = lambdify((EO+PT+pt), F)

    # Define weight matrix
    err = np.dstack((SigX, SigY, SigZ)).reshape(1, -1)  # Error vector
    W = np.matrix(np.diag(s**2 / (err**2).flatten()))

    dX = np.ones(1)                              # Initial value for iteration

    # Iteration process
    while abs(dX.sum()) > 10**-12:
        # Compute coefficient matrix and constants matrix
        A = np.zeros((2 * len(xa), err.shape[1]))
        B = np.array([])
        f = np.array([])

        #  Row and column index which is used to update values of A matrix
        i = 0
        j = 0
        for l in np.dstack((XA, YA, ZA, xa, ya)).reshape(-1, 5):
            val = np.append(np.array(X0).flatten(), l)
            A[i:i+2, j:j+3] = FuncJFl(*val[:-2])
            B = np.append(B, FuncJFx(*val[:-2]))
            f = np.append(f, -FuncF(*val))
            i += 2
            j += 3
        B = np.matrix(B.reshape(-1, 6))
        f = np.matrix(f.reshape(-1, 1))

        Qe = (A * W * A.T).I
        We = Qe.I
        N = (B.T * We * B)                  # Compute normal matrix
        t = (B.T * We * f)                  # Compute t matrix
        dX = N.I * t                        # Compute unknown parameters
        V = W.I * A.T * We * (f - B * dX)   # Compute residual vector

        X0 += dX            # Update initial values

        # Compute error of unit weight
        res = (V.T * W * V)[0, 0]
        s0 = (res / (B.shape[0] - B.shape[1]))**0.5
        # print "Error of unit weight : %.4f" % s0

    # Compute other informations
    SigmaXX = s0**2 * N.I
    paramStd = np.sqrt(np.diag(SigmaXX))
    XL, YL, ZL, Omega, Phi, Kappa = np.array(X0).flatten()

    # Output results
    print "Exterior orientation parameters:"
    print (" %9s %11s %11s") % ("Parameter", "Value", "Std.")
    print " %-10s %11.6f %11.6f" % (
        "Omega(deg)", deg(Omega), deg(paramStd[3]))
    print " %-10s %11.6f %11.6f" % (
        "Phi(deg)", deg(Phi), deg(paramStd[4]))
    print " %-10s %11.6f %11.6f" % (
        "Kappa(deg)", deg(Kappa), deg(paramStd[5]))
    print " %-10s %11.6f %11.6f" % ("XL", XL, paramStd[0])
    print " %-10s %11.6f %11.6f" % ("YL", YL, paramStd[1])
    print " %-10s %11.6f %11.6f" % ("ZL", ZL, paramStd[2])
    print "\nError of unit weight : %.6f" % s0


def main():
    parser = OptionParser(usage="%prog [-i] [-s]", version="%prog 0.1")

    # Define options
    parser.add_option(
        "-i", "--input",
        help="read input data from FILE, the default value is \"input.txt\"",
        metavar="FILE")
    parser.add_option(
        "-s", "--sigma",
        type="float",
        dest="s",
        help="define a priori error, the default value is 5 (deg)",
        metavar="N")

    # Instruct optparse object
    (options, args) = parser.parse_args()

    # Define default values if there are nothing given by the user
    if not options.input:
        options.input = "input.txt"

    if not options.s:
        options.s = rad(5)

    spaceResection(inputFile=options.input, s=options.s)

    return 0


if __name__ == '__main__':
    main()

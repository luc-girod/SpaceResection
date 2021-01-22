# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:27:46 2021

@author: lucg
"""
import numpy as np
import pandas as pd
import gdal
from optparse import OptionParser
from matplotlib import pyplot
import plyfile
import time

def RotMatrixFromAngles(O,P,K):
    
    RX=np.array([[1,0,0],
                 [0,np.cos(O),-np.sin(O)],
                 [0,np.sin(O),np.cos(O)]])    
    RY=np.array([[np.cos(P),0,np.sin(P)],
                 [0,1,0],    
                 [-np.sin(P),0,np.cos(P)]])
    RZ=np.array([[np.cos(K),-np.sin(K),0],
                 [np.sin(K),np.cos(K),0],
                 [0,0,1]])
    
    return RX.dot(RY.dot(RZ))


def XYZ2Im(aPtWorld,aCam,aImSize):
    '''
    Function to project a point in world coordinate into an image

    :param aPtWorld: 3d point in world coordinates
    :param aCam: array describing a camera [position, rotation, focal]
    :param aImSize: Size of the image
    :return:  2d point in image coordinates
    '''
    # World to camera coordinate
    aPtCam=np.linalg.inv(aCam[1]).dot(aPtWorld-aCam[0])
    # Test if point is behind camera (Z positive in Cam coordinates)
    if aPtCam[2]<0:
        return None
    #print("PtCam =", aPtCam)
    # Camera to 2D projected coordinate
    aPtProj=[aPtCam[0]/aPtCam[2],aPtCam[1]/aPtCam[2]]
    #print("PtProj =", aPtProj)
    # 2D projected to image coordinate
    aPtIm=[aImSize[0]/2,aImSize[1]/2]+np.array(aCam[2]).dot(aPtProj)
    if aPtIm[0]>0 and aPtIm[1]>0 and aPtIm[0]<aImSize[0] and aPtIm[1]<aImSize[1]:
        return aPtIm
    else:
        return None


def Raster2Array(raster_file, raster_band=1, nan_value=-9999):
    '''
    Function to convert a raster to an array

    :param raster_file: raster filepath
    :param raster_band: band to export
    :nan_value: value recognized as nan in raster. Default -9999
    :return:  array with the columns X,Y,value.
    '''
    
    # Open reference DEM
    myRaster=gdal.Open(raster_file)
    geot = myRaster.GetGeoTransform()
    Xsize=myRaster.RasterXSize
    Ysize=myRaster.RasterYSize
    data=myRaster.GetRasterBand(raster_band).ReadAsArray(0, 0, Xsize, Ysize)
    data[data==nan_value] = np.nan

    # define extent and resoltuion from geotiff metadata
    extent = [geot[0], geot[0] + np.round(geot[1],3)*Xsize, geot[3], geot[3] + np.round(geot[5],3)*Ysize]

    
    # Create the X,Y coordinate meshgrid
    Xs = np.linspace(extent[0]+np.round(geot[1],3),extent[1], Xsize)
    Ys = np.linspace(extent[2]+ np.round(geot[5],3), extent[3], Ysize)
    XX, YY = np.meshgrid(Xs, Ys)
    
    XYZ = np.vstack((XX.flatten(),YY.flatten(),data.flatten())).T 
    return XYZ


def ProjectImage2DEM(dem_file, image_file, output, aCam, dem_nan_value=-9999):
    '''
    Function to project an image to a DEM

    :param dem_file: DEM raster filepath
    :param image_file: image filepath
    :param output: output point cloud filepath
    :param aCam: array describing a camera [position, rotation, focal]
    '''
    print('aCam=', aCam)
    tic = time.perf_counter()
    aDEM_as_list=Raster2Array(dem_file,nan_value=dem_nan_value)

    toc = time.perf_counter()
    print(f"DEM converted in {toc - tic:0.4f} seconds")
    # Compute the distance between every point in the DEM and the camera
    aDistArray=np.linalg.norm(aDEM_as_list-aCam[0], axis=1)

    # Load in image
    anImage=pyplot.imread(image_file).T
    if len(anImage.shape)==2:
        anImage = np.stack((anImage,anImage,anImage), axis=2)
    # For each pixel in image, store the XYZ position of the point projected to it,
    # and the distance to that point, if a new point would take the same position,
    # keep the closest point
    aXYZinImage=np.zeros([anImage.shape[0],anImage.shape[1],4])*np.nan
    tic = time.perf_counter()
    for i in range(aDEM_as_list.shape[0]):
        aProjectedPoint=XYZ2Im(aDEM_as_list[i],aCam,anImage.shape)
        if not (aProjectedPoint is None):
            aDist=aXYZinImage[int(aProjectedPoint[0]),int(aProjectedPoint[1])][3]
            if np.isnan(aDist) or (not np.isnan(aDistArray[i]) and aDist>aDistArray[i]):
                 aXYZinImage[int(aProjectedPoint[0]),int(aProjectedPoint[1])]=[aDEM_as_list[i][0],aDEM_as_list[i][1],aDEM_as_list[i][2],aDistArray[i]]
                # print(aXYZinImage[int(aProjectedPoint[0]),int(aProjectedPoint[1])])
            #else:
            #    print('Nope ', aDist, aDistArray[i])  
    toc = time.perf_counter()
    print(f"Position of each pixel computed in {toc - tic:0.4f} seconds")
            
    # export ply file
    vertex = np.array([],dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    # export each point X, Y, Z, R, G, B
    for i in range(anImage.shape[0]):
        for j in range(anImage.shape[1]):
            if not np.isnan(aXYZinImage[i,j][3]):
                aPoint=np.array([(aXYZinImage[i,j][0],aXYZinImage[i,j][1],aXYZinImage[i,j][2], anImage[i,j][0],anImage[i,j][1],anImage[i,j][2])],dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
                vertex=np.concatenate((vertex,aPoint), axis=0)
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text=True).write(output)
    print('Total points : ', vertex.shape)
    print('PLY file extracted')
    
    return 0

def main():
    parser = OptionParser(usage="%prog [options]", version="%prog 0.1")

    # Define options
    parser.add_option(
        '-d', '--dem',
        default=None,
        help="read input DEM from FILE",
        metavar='FILE')
    
    parser.add_option(
        '-i', '--image',
        default=None,
        help="read input image from FILE",
        metavar='FILE')

    parser.add_option(
        '-o', '--output',
        default="result.ply",
        help="name of output file, default value is \"result.ply\"",
        metavar='FILE')

    parser.add_option(
        '-c', '--campos',
        type='float',
        dest='C',
        default=None,
        help="defines the camera position",
        metavar='N')
    parser.add_option(
        '-r', '--rot',
        type='float',
        dest='C',
        default=None,
        help="defines the camera rotation matrix",
        metavar='N')
    parser.add_option(
        '-f', '--foc',
        type='float',
        dest='C',
        default=None,
        help="defines the camera's focal lenght (in pixels)'",
        metavar='N')
    # Instruct optparse object
    (options, args) = parser.parse_args()
    
    aCam=[options.campos,options.rot,options.foc];
    
    ProjectImage2DEM(options.dem, options.image, options.output, aCam)

    return 0

# INput Finse
dem_file='E://WebcamFinse//time_lapse_finse_DSM_mid.tif'
image_file='E://WebcamFinse//2019-05-24_12-00.jpg'
output='E://WebcamFinse//output_full.ply'
R=[[-0.25363216,-0.10670329,-0.96139749],[-0.71147276,-0.65278552,0.26014914],[-0.65534513,0.74999032,0.08965091]]
aCam=[[419175.787830,6718422.876705,1217.170495],R,1255]
ProjectImage2DEM(dem_file, image_file, output, aCam, dem_nan_value=1137.75)
R=RotMatrixFromAngles(115,17,75)



#Input Cucza
R=RotMatrixFromAngles(3.3,-4.1,5.5)
# dem_file='E://WebcamFinse//Cucza//DEM.tif'
# image_file='E://WebcamFinse//Cucza//Abbey-IMG_0209.jpg'
# output='E://WebcamFinse//Cucza//output.ply'
# R=[[0.993534882295323163,0.0929006109947966841,0.0652526944977123435],[0.0878277479180877285,-0.993176223631756505,0.0767285833845516713],[0.0719355569802246908,-0.0705015268583363691,-0.994914473888378059]]
# aCam=[[209.89527614679403023,91.20530793577831,107.031846453497209],R,2011.8874387887106]
# ProjectImage2DEM(dem_file, image_file, output, aCam)




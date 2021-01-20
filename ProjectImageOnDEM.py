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

def RotMatrixFromAngles(O,P,K):
    
    RX=np.array([[1,0,0],
                 [0,np.cos(O),-np.sin(O)],
                 [0,np.sin(O),np.cos(O)]])    
    RY=np.array([[np.cos(O),0,np.sin(O)],
                 [0,1,0],    
                 [-np.sin(O),0,np.cos(O)]])
    RZ=np.array([[np.cos(O),-np.sin(O),0],
                 [np.sin(O),np.cos(O),0]],
                 [0,0,1])
    
    return RX.dot(RY.dot(RZ))


def XYZ2Im(aPtWorld,aR,aC,aFoc,aImSize):
    '''
    Function to project a point in world coordinate into an image

    :param aPtWorld: 3d point in world coordinates
    :param aR: rotation matrix
    :param aC: Camera position
    :param aFoc: Focal lenght
    :param aImSize: Size of the image
    :return:  2d point in image coordinates
    '''

    # World to camera coordinate
    aPtCam=np.linalg.inv(aR).dot(aPtWorld-aC)
    #print("PtCam =", aPtCam)
    # Camera to 2D projected coordinate
    aPtProj=[aPtCam[0]/aPtCam[2],aPtCam[1]/aPtCam[2]]
    #print("PtProj =", aPtProj)
    # 2D projected to image coordinate
    aPtIm=aImSize/2+np.array(aFoc).dot(aPtProj)
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


def ProjectImage2DEM(dem_file, image_file, output, aCam):
    '''
    Function to project an image to a DEM

    :param dem_file: DEM raster filepath
    :param image_file: image filepath
    :param output: output point cloud filepath
    :param aCam: array describing a camera [position, rotation, focal]
    '''
    
    aDEM_as_list=Raster2Array(dem_file)
    # Compute the distance between every point in the DEM and the camera
    aDistArray=np.linalg.norm(aDEM_as_list-aCam[0])
    anImage=pyplot.imread(image_file)
    
    # For each pixel in image, store the XYZ position of the point projected to it,
    # and the distance to that point, if a new point would take the same position,
    # keep the closest point
    aXYZinImage=np.zeros([anImage.shape[0],anImage.shape[1],4])*np.nan
    
    for i in range(aDEM_as_list.shape[0]):
        aProjectedPoint=int(XYZ2Im(aDEM_as_list[i],aCam))
        if aProjectedPoint:
            aDist=aXYZinImage[[aProjectedPoint,4]]
            if aDist==np.nan or aDist>aDistArray[i]:
                 aXYZinImage[aProjectedPoint,:]=[aDEM_as_list[i],aDistArray[i]]
            
    # export ply file
    # ply header
    
    # count how many points have values
    
    # export each point X, Y, Z R, G, B
    for i in range(anImage.shape[0]):
        for j in range(anImage.shape[1]):
            if aXYZinImage[i,j][1:3] != np.nan:
                print([aXYZinImage[i,j][1:3], anImage[i,j]])
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


# options.campos=[419175.787830,6718422.876705,1217.170495]
# options.dem='E://WebcamFinse//time_lapse_finse_DSM_mini.tif'
# options.image='E://WebcamFinse//2019-05-24_12-00.jpg'
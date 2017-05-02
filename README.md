SpaceResection
==========

## Description
This repository is an implementation of space resection, which can get the position and attitude of an camera with given information.
The necessary information include:  
+ focal length [mm]  
+ image coordinates [mm]  
+ object point coordinates [m]  

There must be more than three point observations to solve six unknown parameters.

## Usage
First, create an input file. The format of an input file must like:
```
<f>
<P-Name> <x1> <y1> <X1> <Y1> <Z1> <X1-Err> <Y1-Err> <Z1-Err>
<P-Name> <x2> <y2> <X2> <Y2> <Z2> <X2-Err> <Y2-Err> <Z2-Err>
<P-Name> <x3> <y3> <X3> <Y3> <Z3> <X3-Err> <Y3-Err> <Z3-Err>
...
```
`<f>` is the focal length of camera.  
`<P-Name>` is the point name.  
`<x*> <y*>` stands for the image point coordinates with corresponding object point coordinates `<X*> <Y*> <Z*>`.  
Since the object point coordinates are treated as observables with uncertainty, so they must have errors `<X*-Err> <Y*-Err> <Z*-Err>`.

Then you can just call `./spaceResecion.py -i <input file>` to start the computation.  
You can also type `./spaceResecion.py -h` for more information about this repository.  
There are already two input files serve as an example.


## Requirements

### Python
[Python v2.7.X](https://www.python.org) with the following modules to be installed.

-[Numpy](http://www.numpy.org)  
-[Sympy](http://www.sympy.org/en/index.html)  
-[Pandas](http://pandas.pydata.org/)  

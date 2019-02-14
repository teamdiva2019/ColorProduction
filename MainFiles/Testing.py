import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset
import datetime as dt

# Need this line to run the file from anywhere
sys.path.append('.//MainFiles')
from MakeAndSavePUMap import makeColorMap
from WriteToFGA import writeFGAFile
from Metadata import Metadata

filename = '..\\..\\Data\\wind10m200309.nc'
writeFGAFile(Dataset(filename), 10, 2, padWidth=1)

# x = [0,0,1,1]
# y = [0,1,0,1]
# u = [0.8,-0.1,0,1]
# v = [0.2,0.9,1,1]
#
# plt.quiver(x, y, u, v)
# plt.show()
#
# topLeft = np.array([-0.1,0.9])
# topRight = np.array([1,1])
# bottomLeft = np.array([0.8,0.2])
# bottomRight = np.array([0,1])
#
# interpLoc = np.array([0.2,0.6])
# weightX = 0.2 / (1 - 0)
# weightY = 0.6 / (1 - 0)
# # Find the vectors directly above and below the
# # intended location by interpolating between
# # the top horizontal sides
# interAbove = (1-weightX) * topLeft + weightX * topRight
# interBelow = (1-weightX) * bottomLeft + weightX * bottomRight
# print(interAbove, interBelow)
# x.extend([0.2,0.2])
# y.extend([0,1])
# u.extend([interBelow[0], interAbove[0]])
# v.extend([interBelow[1], interAbove[1]])
# plt.quiver(x,y,u,v)
# plt.show()
#
# # Now we interpolate from interBelow to interAbove according to
# # weightY
# resVector = (1-weightY) * interBelow + weightY * interAbove
# x.append(0.2)
# y.append(0.6)
# u.append(resVector[0])
# v.append(resVector[1])
# plt.quiver(x,y,u,v)
# plt.show()

sys.exit(1)
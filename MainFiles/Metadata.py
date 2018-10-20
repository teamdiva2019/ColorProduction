#######################################
# File: Metadata.py
# Location: ColorProduction/MainFiles
# Author: Mughil Muthupari
#
#
# This file holds the class that represents
# all of the metadata in a netcdf file.
# Main functions include writing metadata to a file
# and loading data from set bounds. Should
# clean up code in other parts.
#######################################

from netCDF4 import Dataset, num2date
import numpy as np
import sys
import os

# If both are given, then it's [start, end)
# If only startTime is given, then it's to the end.
# If only endTime is given, then it's from the beginning.
# Giving a timeStamp overrides everything

class Metadata:
    def __init__(self, filename, startTime=None,
                 endTime=None, timeStamp=None):
        self.sT = startTime
        self.eT = endTime
        self.tS = timeStamp
        self.fn = filename
        self.dataset = Dataset(self.fn)
        self.allVariables = self.dataset.variables
        self.varNames = list(filter(lambda x: 'bnds' not in x, list(self.allVariables.keys())))
        # Remove the dimensions
        self.varNames = list(filter(lambda x: x not in self.dataset.dimensions, self.varNames))
        self.isVectorized = False
        if len(self.varNames) > 1:
            self.isVectorized = True
        else:
            self.var = self.varNames[0]

    # Loads the data
    def loadData(self):
        data = np.stack([self.allVariables[var][:].flatten() for var in self.varNames])
        if isinstance(data, np.ma.core.MaskedArray):
            data = data.filled(np.nan)
        return np.linalg.norm(data, axis=0)

    # Following function writes metadata to
    # a file and returns the loaded data.
    def writeMetadata(self):
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        with open(scriptDir + './/metadata.txt', 'w') as f:
            # For each variable, write the name of it
            f.write('Variables\n')
            for var in self.varNames:
                f.write(var + '\n' + self.allVariables[var].long_name +
                        '\n' + self.allVariables[var].units + '\n')
            # Read in the data. No distinction here on whether 1D or 2D
            # Stack x1, x2, ..., xn coordinates on top of each other
            # data = np.stack([self.allVariables[var][:].flatten() for var in self.varNames])
            # if isinstance(data, np.ma.core.MaskedArray):
            #     data = data.filled(np.nan)
            # Only need to print shape once, as all variables need to have
            # same shape
            f.write('Shape\n')
            dataShape = list(self.allVariables[self.varNames[0]].shape)
            f.write(str(dataShape) + '\n')
            # Get the dimensions of a sample variable
            # By design, the other variables must have the
            # same dimensions
            varDims = self.allVariables[self.varNames[0]].dimensions
            # For each dimension, write the name,
            # the start value, the end value,
            # and the step
            f.write('Dimensions\n')
            for dim in varDims:
                vals = self.allVariables[dim][:]
                if dim == 'time':
                    timeUnit = self.allVariables[dim].units
                    try:
                        timeCalendar = self.allVariables[dim].calendar
                    except AttributeError:
                        timeCalendar = u"gregorian"
                    endpoints = num2date([vals[0], vals[-1]], units=timeUnit, calendar=timeCalendar)
                    timeStep = num2date(vals[1], units=timeUnit, calendar=timeCalendar) - endpoints[0]
                    f.write('\n'.join(map(str, [dim, endpoints[0], endpoints[1], timeStep])) + '\n')
                else:
                    f.write('\n'.join(map(str, [dim, vals[0], vals[-1], vals[1] - vals[0]])) + '\n')


meta = Metadata('..//..//Data//pressfc201012.nc')
meta.writeMetadata()

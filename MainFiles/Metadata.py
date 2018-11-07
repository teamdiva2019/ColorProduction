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
import datetime as dt
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
        self.sTIndex = 0
        self.eTIndex = -1
        self.tS = timeStamp
        self.tSIndex = None
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
        # Prioritize having a single timestamp.
        # Otherwise, use start and end times.
        # If the end time has -1, that implies a start time
        # has not been specified, and so get everything until the end.
        # Otherwise, include the end time in the range.
        if self.tSIndex is not None:
            data = np.stack([self.allVariables[var][self.tSIndex].flatten() for var in self.varNames])
        else:
            data = np.stack([self.allVariables[var][self.sTIndex : self.eTIndex + 1].flatten()
                             for var in self.varNames])
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
            # f.write('Shape\n')
            dataShape = list(self.allVariables[self.varNames[0]].shape)
            # f.write(str(dataShape) + '\n')
            # Get the dimensions of a sample variable
            # By design, the other variables must have the
            # same dimensions
            varDims = self.allVariables[self.varNames[0]].dimensions
            # For each dimension, write the name,
            # the start value, the end value,
            # and the step
            dimstr = 'Dimensions\n'
            for dim in varDims:
                vals = self.allVariables[dim][:]
                if dim == 'time':
                    timeUnit = self.allVariables[dim].units
                    try:
                        timeCalendar = self.allVariables[dim].calendar
                    except AttributeError:
                        timeCalendar = u"gregorian"
                    # Convert everything to datetime
                    vals = num2date(vals, units=timeUnit, calendar=timeCalendar)
                    timeStep = vals[1] - vals[0]
                    # First check if a timestamp is expected
                    if self.tS is not None:
                        if isinstance(self.tS, dt.datetime):
                            self.tSIndex = np.argmin(np.abs(self.tS - vals))
                        else:
                            self.tSIndex = self.tS
                        # Recalculate the closest timestamp from the index now...
                        self.tS = vals[self.tSIndex]
                        dimstr += '\n'.join(map(str, [dim, self.tS, self.tS, timeStep])) + '\n'
                    else:
                        # Find the index for the closest start time
                        if self.sT is not None:
                            # Can either be an index or straight up datetime
                            if isinstance(self.sT, dt.datetime):
                                self.sTIndex = np.argmin(np.abs(self.sT - vals))
                            else: # Assumed to have already checked for the proper type
                                self.sTIndex = self.sT
                        if self.eT is not None:
                            if isinstance(self.eT, dt.datetime):
                                self.eTIndex = np.argmin(np.abs(self.eT - vals))
                            else:
                                self.eTIndex = self.eT
                                if self.eTIndex < 0:
                                    self.eTIndex = len(vals) - 1
                        dimstr += '\n'.join(map(str, [dim, vals[self.sTIndex], vals[self.eTIndex],
                                                      timeStep])) + '\n'
                else:
                    dimstr += '\n'.join(map(str, [dim, vals[0], vals[-1], vals[1] - vals[0]])) + '\n'
            # Append the shape string based on time bounds...
            # Assume only time gets changed...
            if self.tS is not None:
                dataShape[0] = 1
            else:
                dataShape[0] = self.eTIndex - self.sTIndex + 1
            f.write('Shape\n')
            f.write(str(dataShape) + '\n')
            f.write(dimstr)


# meta = Metadata('..//..//Data//pressfc201012.nc', startTime=0, endTime=-1)
# meta.writeMetadata()

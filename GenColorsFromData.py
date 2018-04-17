import numpy as np
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset, num2date
import GenColorMap
import time
import sys
import struct
import os

# The file takes a netcdf file and generates a
# binary file with each of the required colors.
# If the data is vectorized, it will also
# generate a directions file.
# BY DEFAULT: The file writing is turned off
# because that is main thing that slows down the program.
# Turn on at your own risk.


# Get script directory
scriptDir = os.path.dirname(os.path.realpath(__file__))
# Read file path
filename = ''
try:
    filename = sys.argv[1]
except IndexError:
    print('No file name!')
    sys.exit(1)
data = Dataset(filename)
allVariables = data.variables
# print(allVariables)
# Sometimes we have time_bnds, lat_bnds, etc.
# Keep anything that doesn't have 'bnds'
varNames = list(filter(lambda x: 'bnds' not in x, list(allVariables.keys())))
# Remove the dimensions
varNames = list(filter(lambda x: x not in data.dimensions, varNames))
print(varNames)
isVectorized = False
if len(varNames) > 1:
    isVectorized = True
firstVar = varNames[0]
with open(scriptDir + './/metadata.txt', 'w') as f:
    # For each variable, write the name of it
    f.write('Variables\n')
    for var in varNames:
        f.write(var + '\n' + allVariables[var].long_name +
                '\n' + allVariables[var].units + '\n')
    # Read in the data. No distinction here on whether 1D or 2D
    # Stack x1, x2, ..., xn coordinates on top of each other
    data = np.stack([allVariables[var][:].flatten() for var in varNames])
    if isinstance(data, np.ma.core.MaskedArray):
        data = data.filled(np.nan)
    # Directions for 2 variables ONLY!!!
    if isVectorized:
        if len(varNames) == 2:
            directions = np.arctan2(data[1], data[0])  # arctan of y/x, so data[1] is first
        else:
            print('Directions in 3D space not supported!')
            sys.exit(2)
    # Find magnitude
    data = np.linalg.norm(data, axis=0)
    # Print maximum and minimum
    minVal = np.nanmin(data)
    maxVal = np.nanmax(data)
    f.write(str(minVal) + ' ' + str(maxVal) + '\n')

    # Only need to print shape once, as all variables need to have
    # same shape
    f.write('Shape\n')
    dataShape = list(allVariables[firstVar].shape)
    f.write(str(dataShape) + '\n')
    # Get the dimensions of a sample variable
    # By design, the other variables must have the
    # same dimensions
    varDims = allVariables[firstVar].dimensions
    # For each dimension, write the name,
    # the start value, the end value,
    # and the step
    f.write('Dimensions\n')
    for dim in varDims:
        vals = allVariables[dim][:]
        if dim == 'time':
            timeUnit = allVariables[dim].units
            try:
                timeCalendar = allVariables[dim].calendar
            except AttributeError:
                timeCalendar = u"gregorian"
            endpoints = num2date([vals[0], vals[-1]], units=timeUnit, calendar=timeCalendar)
            timeStep = num2date(vals[1], units=timeUnit, calendar=timeCalendar) - endpoints[0]
            f.write('\n'.join(map(str, [dim, endpoints[0], endpoints[1], timeStep])) + '\n')
        else:
            if 'lat' in dim:
                latBounds = (vals[0], vals[-1])
            elif 'lon' in dim:
                lonBounds = (vals[0], vals[-1])
            f.write('\n'.join(map(str, [dim, vals[0], vals[-1], vals[1] - vals[0]])) + '\n')

start = time.clock()
# The second argument is either the
# name of the color map, which may be followed by color
# stops if the map isn't available
# to generate a perceptually uniform color map

colorMap = ''
try:
    colorMap = sys.argv[2]
except IndexError:
    print('No color map!')
    sys.exit(1)
# colorMap = 'viridis'
# Check if the color map exists
if not os.path.isfile(scriptDir + './/ColorMaps//' + colorMap + '.txt'):
    # Make the colormap in another file
    GenColorMap.runScript(sys.argv[2:])

# Read the colormap
PUcols = np.loadtxt(scriptDir + './/ColorMaps//' + colorMap + '.txt')

# Create the Color map object
gendMap = ListedColormap(PUcols, N=len(PUcols))
# Set nan values as black
gendMap.set_under('black', 1)

################################ WRITE TO FGA ##############################

# # Writes the directions to a Fluid-Grid ASCII (.fga) file.
# # The third dimension will be time i.e. going higher
# # in z is going forward in time. All these
# # vectors are 2D. That means that the last column will always be 0.
# # Assume that the dataShape goes (time, lat, lon)
# if len(varNames) > 0:
#     minBox = (latBounds[1], lonBounds[0], 0)
#     maxBox = (latBounds[0], lonBounds[1], dataShape[0])
#     # Create the 2D array of vectors
#     vectors = np.zeros((len(directions), 3), dtype='float32')
#     # The first column is x -> cosine, The second column is y -> sine
#     vectors[:, 0] = np.cos(directions)
#     vectors[:, 1] = np.sin(directions)
#     # Delete the directions array, we don't need it anymore
#     del directions
#     with open(scriptDir + './/Directions//directs_' + firstVar + '.fga', 'wb') as f:
#         # The resolution is the same as the size, except time is pushed to the back
#         dataShape.append(dataShape.pop(0))
#         np.savetxt(f, np.array(dataShape))
#         print('Wrote resolution.')
#         # The min and max box are the in-house coordinates of the data
#         np.savetxt(f, np.array(minBox))
#         np.savetxt(f, np.array(maxBox))
#         print('Wrote bounding box coordinates.')
#         # Now to write each vector
#         np.savetxt(f, vectors)
#         print('Wrote vector data.')

########################## WRITING COLORS ##########################

# Normalize the data and create a ScalarMappable object.
# This allows us to efficiently map the values to colors
norm = matplotlib.colors.Normalize(vmin=minVal, vmax=maxVal)
mapper = cm.ScalarMappable(norm=norm, cmap=gendMap)

# Map the data. Take the first 3 columns as last column is alpha
colorMappedData = mapper.to_rgba(data, alpha=False, bytes=True)[:, :3]

# Save the data to a binary file to minimize size
# colorMappedData.tofile(scriptDir + './/ColorFromData//colors_' + firstVar + '.bin')

end = time.clock()
print(end - start, 'seconds.')

import numpy as np
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset, num2date
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
                latBounds = np.array([vals[0], vals[-1], vals[1] - vals[0]])
            elif 'lon' in dim:
                lonBounds = np.array([vals[0], vals[-1], vals[1] - vals[0]])
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

# Writes the directions to a Fluid-Grid ASCII (.fga) file.
# The third dimension will be time i.e. going higher
# in z is going forward in time. All these
# vectors are 2D. That means that the last column will always be 0.
# Assume that the dataShape goes (time, lat, lon)
if len(varNames) > 0:
    print('Magnitude:', data)
    print('Directions:', directions)
    oneLevelPoints = dataShape[1] * dataShape[2]  # Working with one level for now
    # Convert the degree step to radians
    # makes it easier later
    latBounds *= np.pi / 180
    lonBounds *= np.pi / 180
    print('Lat bounds:', latBounds)
    print('Long bounds:', lonBounds)
    # I'll be following the word doc and mapping
    scale = 1  # Change this to affect the size of the sphere
    minBox = [(-1) * scale] * 3
    maxBox = [scale] * 3
    # Distance between vectors on top latitude.
    # Change this to adjust resolution
    distBet = 1
    # Radius of the top intersecting sphere
    rTop = distBet / lonBounds[2]
    # Diameter of sphere in resolution units i.e. not the actual diameter
    D = int(2 * rTop / np.sin(latBounds[2]) + 1)
    resStep = (2 * scale) / D
    print(D, resStep)
    # Calculate orientation of vectors
    # Start with unit vectors
    uvs = np.array([np.cos(directions[:oneLevelPoints]), np.sin(directions[:oneLevelPoints])])\
        .reshape(dataShape[1], dataShape[2], 2)
    vects3D = np.zeros(dataShape[1] * dataShape[2] * 3).reshape(dataShape[1], dataShape[2], 3)
    vects3D[:, :, :2] = uvs
    del uvs
    print(vects3D[0, 0])
    for latInd in range(dataShape[1]):
        for lonInd in range(dataShape[2]):
            lat = latBounds[0] + latInd * latBounds[2]
            lon = lonBounds[0] + lonInd * lonBounds[2]
            rotMatrix = np.dot(
                np.array([
                    [np.cos(lon), -np.sin(lon), 0],
                    [np.sin(lon), np.cos(lon), 0],
                    [0, 0, 1]
                ]),
                np.array([
                    [np.cos(lat), 0, np.sin(lat)],
                    [0, 1, 0],
                    [-np.sin(lat), 0, np.cos(lat)]
                ])
            )
            # Rotate the vector at the location
            vects3D[latInd, lonInd] = np.dot(rotMatrix, vects3D[latInd, lonInd])
    print(vects3D[0, 0])

    # Now we will deal with placing the vectors at the correct location
    fgaVectors = np.zeros((D, D, D, 3))
    for latInd in range(dataShape[1]):
        for lonInd in range(dataShape[2]):
            lat = latBounds[0] + latInd * latBounds[2]
            lon = lonBounds[0] + lonInd * lonBounds[2]
            x, y, z = (
                (D/2) * np.sin(np.pi/2 - lat) * np.cos(lon),
                (D/2) * np.sin(np.pi/2 - lat) * np.cos(lat),
                (D/2) * np.cos(np.pi/2 - lat)
            )
            # Find the index
            xi, yi, zi = (
                np.round((x + (D/2)) / resStep),
                np.round((y + (D/2)) / resStep),
                np.round((z + (D/2)) / resStep)
            )
            # Set it equal to it in our fga array
            fgaVectors[xi, yi, zi] = vects3D[latInd, lonInd]
    print(fgaVectors)
    # with open(scriptDir + './/Directions//directs_' + firstVar + '.fga', 'wb') as f:
    #     np.savetxt(f, vectors, delimiter=',', newline='\r\n', fmt='%4.7f')

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

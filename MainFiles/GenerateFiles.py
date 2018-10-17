import argparse
import os
import sys
import numpy as np
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset, num2date

# Need this line to run the file from anywhere
sys.path.append('.//MainFiles')
from MakeAndSavePUMap import makeColorMap
from WriteToFGA import writeFGAFile

def isNCFile(filename):
    if filename[-3:] != '.nc':
        raise argparse.ArgumentTypeError('{} is an invalid file name (only netCDF files allowed)'
                                         .format(filename))
    return filename
def isValidColor(val):
    if val is not int and 0 > int(val) > 255:
        raise argparse.ArgumentTypeError('{} is an invalid color value (0-255 inclusive)'
                                         .format(val))
    return int(val)

# Get script directory
scriptDir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(scriptDir))

parser = argparse.ArgumentParser(description='This program generates the appropriate color files'
                                             'and FGA files for loading into Unreal Engine.')
parser.add_argument('infile', type=isNCFile, help='NetCDF file to be transformed to color')
parser.add_argument('colormap', type=str, default='inferno',
                    help='Color map to use for transformation')
parser.add_argument('-c', '--colors', type=isValidColor,
                    required=False, dest='colors', nargs='+',
                    help='A list of gradients for the color map. Specify a list of 3'
                         '0-255 integers (RGB) for each color gradient. Requires at least'
                         'two colors to generate a color map.')
parser.add_argument('-fga', '--FGA', required=False,
                    dest='rad_step', nargs=2,
                    help='Specify the RADIUS then the resolution STEP for an FGA'
                         'file built from vectorized file. Please specify the radius'
                         'FIRST and then the step')

args = parser.parse_args(sys.argv[1:])
isVectorized = False
if args.rad_step is not None:
    isVectorized = True
    rad = args.rad_step[0]
    step = args.rad_step[1]

# Check to see if the list of colors is valid
if args.colors is not None:
    if len(args.colors) % 3 != 0:
        raise argparse.ArgumentTypeError('Incorrect number of colors given.')
    # Otherwise, reshape into an n x 3 array and produce the colormap
    args.colors = np.reshape(args.colors, (-1, 3))
    makeColorMap(args.colormap, args.colors)
# Once it's guaranteed there's a color map, then generate the colors
#################### METADATA #####################
data = Dataset(args.infile)
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
# Read the colormap
PUcols = np.loadtxt(scriptDir + './/ColorMaps//' + args.colormap + '.txt')

# Create the Color map object
gendMap = ListedColormap(PUcols, N=len(PUcols))
# Set nan values as black
gendMap.set_under('black', 1)

# Normalize the data and create a ScalarMappable object.
# This allows us to efficiently map the values to colors
norm = matplotlib.colors.Normalize(vmin=minVal, vmax=maxVal)
mapper = cm.ScalarMappable(norm=norm, cmap=gendMap)

# Map the data. Take the first 3 columns as last column is alpha
colorMappedData = mapper.to_rgba(data, alpha=False, bytes=True)[:, :3]

# Save the data to a binary file to minimize size
colorMappedData.tofile(scriptDir + './/ColorFromData//colors_' + firstVar + '.bin')

# See if we need to make an fga file
if isVectorized:
    writeFGAFile(Dataset(args.infile), rad, step)
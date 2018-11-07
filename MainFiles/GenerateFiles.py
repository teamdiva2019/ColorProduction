import argparse
import os
import sys
import numpy as np
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset
import datetime as dt

# Need this line to run the file from anywhere
sys.path.append('.//MainFiles')
from MakeAndSavePUMap import makeColorMap
# from WriteToFGA import writeFGAFile
from Metadata import Metadata
from MapVectorsTo3D import writeFGAFile

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
def isCorrectDateFormat(val):
    try:
        val = dt.datetime.strptime(val, '%Y-%m-%d-%H-%M-%S')
    except ValueError:
        raise argparse.ArgumentTypeError('{} is an valid date format. Need YYYY-MM-DD HH:MM:SS'
                                         .format(val))
    return val

# Get script directory
scriptDir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(scriptDir))

parser = argparse.ArgumentParser(description='This program generates the appropriate color files '
                                             'and FGA files for loading into Unreal Engine.')
parser.add_argument('infile', type=isNCFile, help='NetCDF file to be transformed to color')
parser.add_argument('colormap', type=str, default='inferno',
                    help='Color map to use for transformation')
parser.add_argument('-c', '--colors', type=isValidColor,
                    required=False, dest='colors', nargs='+',
                    help='A list of gradients for the color map. Specify a list of 3 '
                         '0-255 integers (RGB) for each color gradient. Requires at least '
                         'two colors to generate a color map.')
parser.add_argument('-fga', '--FGA', required=False, type=float,
                    dest='rad_step', nargs=2,
                    help='Specify the RADIUS then the resolution STEP for an FGA '
                         'file built from vectorized file. Please specify the radius '
                         'FIRST and then the step')
parser.add_argument('-st', '--starttime', required=False, type=isCorrectDateFormat,
                    dest='start_time',
                    help='Specify the start time of the data in YYYY-MM-DD-HH-MM-SS.')
parser.add_argument('-et', '--endtime', required=False, type=isCorrectDateFormat,
                    dest='end_time',
                    help='Specify the end time of the data in YYYY-MM-DD-HH-MM-SS.')
parser.add_argument('-ts', '--timestamp', required=False, type=isCorrectDateFormat,
                    dest='time_stamp',
                    help='Specify a certain time stamp to get data from in '
                         'YYYY-MM-DD HH:MM:SS. OVERRIDES start and end time '
                         'arguments.')

args = parser.parse_args(sys.argv[1:])
isVectorized = False
if args.rad_step is not None:
    isVectorized = True
    rad = int(args.rad_step[0])
    step = args.rad_step[1]

# Check to see if the list of colors is valid
if args.colors is not None:
    if len(args.colors) % 3 != 0:
        raise argparse.ArgumentTypeError('Incorrect number of colors given.')
    # Otherwise, reshape into an n x 3 array and produce the colormap
    args.colors = np.reshape(args.colors, (-1, 3))
    makeColorMap(args.colormap, args.colors)
# Check the arguments for times...
startTime = 0
endTime = -1
timestamp = None
if args.time_stamp is not None:
    timestamp = args.time_stamp
elif args.start_time is not None:
    startTime = args.start_time
if args.end_time is not None:
    endTime = args.end_time
# Once it's guaranteed there's a color map, then generate the colors
#################### METADATA #####################
meta = None
if args.time_stamp is not None:
    meta = Metadata(args.infile, timeStamp=timestamp)
else:
    meta = Metadata(args.infile, startTime=startTime, endTime=endTime)
meta.writeMetadata()
data = meta.loadData()
minVal = np.nanmin(data)
maxVal = np.nanmax(data)

# Read the colormap
PUcols = np.loadtxt(os.path.join(scriptDir, '..//ColorMaps//' + args.colormap + '.txt'))

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
colorMappedData.tofile(os.path.join(scriptDir, '..//ColorFromData//colors_' + meta.varNames[0] + '.bin'))

# See if we need to make an fga file
if isVectorized:
    writeFGAFile(Dataset(args.infile), rad, step)
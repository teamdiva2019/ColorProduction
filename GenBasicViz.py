import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# This file will generate a static image of the very
# first time step in the data. It will utilize
# metadata.txt to gather information.

# It should only be used for debugging and not intended
# to be released in the full version.

# NOTE: This file assumes that the dimensions go in the
# order of time, plev, lat, long for simplicity.

# Grab script directory for saving figure later
scriptDir = os.path.dirname(os.path.realpath(__file__))

with open('metadata.txt', 'r') as f:
    content = [line.strip('\n') for line in f]
# Variables starts off. The name right after 'Variables'
# show how the colors and directions files are named.
filename = content[1]
indexOfShape = content.index('Shape')
isVectorized = False
if indexOfShape > 5:  # More than 5 away from 'Variables'
    isVectorized = True
shape = tuple(map(int, content[indexOfShape + 1][1:-1].split(', ')))

# Read in the colors, and gather the 1st time step. If
# there is a pressure level, it would include
# only the first pressure level. This would
# be the first lat * long elements in the binary file.
colors = np.fromfile('.//ColorFromData//colors_' + filename + '.bin',
                     dtype=np.uint8, count=shape[-2] * shape[-1] * 3).reshape(shape[-2], shape[-1], 3)
# Read in directions, if there are any
if isVectorized:
    directions = np.fromfile('.//Directions//directs_' + filename + '.bin',
                             dtype=np.float32, count=shape[-2] * shape[-1]).reshape(shape[-2], shape[-1])
    print(directions)
print(colors)

# Gather the latitude, longitude, pressure, and time data
latData = (float(content[-7]), float(content[-6]), float(content[-5]))
longData = (float(content[-3]), float(content[-2]), float(content[-1]))
timeMin = content[content.index('time') + 1]
print(timeMin)
plevMin = None
if 'plev' in content:
    plevMin = content[content.index('plev') + 1]
# Find the (0, 0) mark. This helps to orient the plot correctly
# Find it using the minimum and the step value
latZero, longZero = np.abs([latData[0] / latData[2], longData[0] / longData[2]]).astype(int)
print(latZero, longZero)
print(latData, longData)
# Plot the image along with correct axes
if isVectorized:
    fig = plt.figure(figsize=(70, 70)) # Need to be able to zoom in
    plt.rcParams.update({'font.size': 32})
else:
    fig= plt.figure(figsize=(20, 20))
    plt.rcParams.update({'font.size': 22})
ax = plt.subplot(111)
if isVectorized:
    ax.quiver(np.cos(directions), np.sin(directions))  # We have directions, so do sin and cos
plt.imshow(colors, origin=[46, 0])
# Label the axes correctly. The origin argument ensures
# correct orientation.
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(
        lambda x, i: "{0:.2f}".format(x + longData[2] + longData[0])
    )
)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(
        lambda y, i: "{0:.2f}".format(y * latData[2] + latData[0])
    )
)
# Label axes and put title.
# Axes are assumed to be degrees North for latitude
# and degrees East for longitude unless otherwise changed.
# Title is taken from the type of data and merged with the
# first time and plev step extracted earlier.
plt.title(content[2] + ' ' + timeMin)
plt.xlabel('Degrees East')
plt.ylabel('Degrees North')
plt.savefig(scriptDir + './/GennedSamples//' + filename + '.png')
plt.show()


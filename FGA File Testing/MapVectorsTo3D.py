import time
import numpy as np
import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset
import sys

# Function to find the closest 2 points to another point
def find2Points(p, pointSpace, vectSpace):
    ds = np.linalg.norm(pointSpace - p, axis=1)
    ls = ds.argsort()[:2]
    return pointSpace[ls], vectSpace[ls], ds[ls]

# Get script directory
scriptDir = os.path.dirname(os.path.realpath(__file__))

# Start the clock
start = time.perf_counter()

# Set the size of the box and
# the resolution step...user inputs.
# Also calculate diameter in res units
boxRad = 5
padWidth = 1
resStep = 0.1
DRes = int(2 * boxRad / resStep + 1)
# Set scaling values
windScale = 40
gravScale = 40

# Read in file
filename = ''
try:
    filename = sys.argv[1]
except IndexError:
    print('No file name!')
    sys.exit(1)
data = Dataset(filename)
allVariables = data.variables

# Sometimes we have time_bnds, lat_bnds, etc.
# Keep anything that doesn't have 'bnds'
varNames = list(filter(lambda x: 'bnds' not in x, list(allVariables.keys())))
# Remove the dimensions
varNames = list(filter(lambda x: x not in data.dimensions, varNames))

latPoints = allVariables['latitude'][:] * np.pi / 180
latStep = latPoints[1] - latPoints[0]

lonPoints = allVariables['longitude'][:] * np.pi / 180
lonStep = lonPoints[1] - lonPoints[0]

numLatPoints = len(latPoints)
numLonPoints = len(lonPoints)

print('Latitude Points:', latPoints)
print('Longitude Points:', lonPoints)

# Set up directions. All will be unit vector to the right (1, 0)
# directions = np.zeros((numLatPoints, numLonPoints))
# u, v = np.cos(directions), np.sin(directions)
u, v = allVariables[varNames[0]][0], allVariables[varNames[1]][0]
##### PLOT THE FLAT VECTORS
x, y = lonPoints * 180 / np.pi, latPoints * 180 / np.pi
plt.quiver(x, y, u, v)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title('Wind Vectors (flat)')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.show()

################# TRANSFORM TO 3D VECTORS ##################
print('Transforming to 3D vectors...')
vects3D = np.zeros((numLatPoints, numLonPoints, 3))
vects3D[:, :, 0] = u
vects3D[:, :, 1] = v

# Make matrix of exact locations of the vectors
exactVectLocs = np.zeros((numLatPoints, numLonPoints, 3))

### Rotate each 2D vector to its 3D counterpart
for i, lat in enumerate(latPoints, 0):
    for j, lon in enumerate(lonPoints, 0):
        # Find the rotation matrix...look up on Wikipedia
        # It is a pair of two transformations, one about
        # the z-axis for proper latitude orientation,
        # and one about the y-axis for proper
        # longitude orientation.
        rotMatrix = np.dot(
            [
                [np.cos(lon), -np.sin(lon), 0],
                [np.sin(lon), np.cos(lon), 0],
                [0, 0, 1]
            ],
            # pi / 2 - lat gives you phi
            [
                [np.cos(np.pi/2 - lat), 0, np.sin(np.pi/2 - lat)],
                [0, 1, 0],
                [-np.sin(np.pi/2 - lat), 0, np.cos(np.pi/2 - lat)]
            ]
        )
        # Before we rotate, we need to convert to a "top-down" view,
        # so the +x-axis will point down and the +y-axis will point right.
        # (x,y,0) ==> (-y,x,0)
        np.dot(rotMatrix, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), out=rotMatrix)
        # Apply the entire rotation matrix
        vects3D[i, j] = np.dot(rotMatrix, vects3D[i, j])

        ### Calculate the 2d point. Basically spherical coordinates
        exactVectLocs[i, j] = boxRad * np.array([
            np.sin(np.pi / 2 - lat) * np.cos(lon),
            np.sin(np.pi / 2 - lat) * np.sin(lon),
            np.cos(np.pi / 2 - lat)
        ])
print('Transform complete!')

# It should be trivial that the closest 2 defined points
# to another point will be on the same xy-plane. Therefore,
# we can cycle through each latitude level and interpolate
# around the longitudes. The interpolation will be linear
# in the direction i.e. a weighted average.
countNonzero = 0
fgaVectors = np.zeros((DRes, DRes, DRes, 3))
print('Interpolating vectors...')
for i, lat in enumerate(latPoints, 0):
    exactLats = exactVectLocs[i]
    vects3DLats = vects3D[i]
    # Now, we need to find the points on this
    # latitude level that are the closest to the sphere
    xyz = np.zeros((DRes ** 2, 3))
    # Make an xy-plane version, since z is constant
    xy = np.mgrid[
        -boxRad: boxRad + 1e-10: resStep,
        -boxRad: boxRad + 1e-10: resStep,
    ].reshape(2, -1).T
    xyz[:, :2] = xy
    xyz[:, 2] = boxRad * np.cos(np.pi/2 - lat)
    xyz = xyz.reshape((DRes, DRes, 3))
    # Calculate distance from each point in this plane
    # to the origin. Anything that is more than the
    # square of resStep gets filtered out.
    dists = np.abs(np.linalg.norm(xyz, axis=2) - boxRad)
    xyz = xyz[np.where(dists < resStep)]
    # Now, for each point in xyz, we find the two defined points
    # that are closest and interpolate between those two points.
    for point in xyz:
        # Get the closest points, the vectors
        # at those points, and the distances
        locs, vects, dists = find2Points(point, exactLats, vects3DLats)
        # Calculate a weighted average of the
        # first vector to the second vector
        weights = 1 - dists / np.sum(dists)
        # Two things we need to calculate:
        # the direction and the magnitude.
        # The magnitude of our vectors are all
        # one for testing, but actual data is not.

        # For direction, we can calculate
        # the angle between the two vectors,
        # since we're in an xy-plane at a z.
        thetaDiff = np.arccos(np.dot(vects[0], vects[1]) /
                          (np.linalg.norm(vects[0]) * np.linalg.norm(vects[1])))
        rotTheta = weights[1] * thetaDiff # Go this much of the way
        # rotTheta just tells magnitude, but does not tell
        # the direction of the turn. This is in the following:
        if vects[0, 0] * vects[1, 1] - vects[0, 1] * vects[1, 0] < 0:
            rotTheta *= -1  # Rotate towards the right, otherwise left
        # Magnitude - weighted average
        magnitude = np.dot(weights, np.linalg.norm(vects, axis=1))
        # Take our first vector, rotate it by rotTheta
        # about the z-axis (because in the same xy-plane)
        # and rescale it by newMag. Rotation matrix time!
        newVector = np.dot([
            [np.cos(rotTheta), -np.sin(rotTheta), 0],
            [np.sin(rotTheta), np.cos(rotTheta), 0],
            [0, 0, 1]
        ], vects[0])

        # OKAY we have our new vector, and where we need to place it.
        # Now, we need to find the point in our box space that is
        # closest to our actual point. Thankfully, because
        # we computed our point space based upon the resolution,
        # it will match up exactly. It's basically an
        # arithmetic progression
        xi, yi, zi = tuple(np.array((point + boxRad) / resStep, dtype=int))   # Really - (-boxRad)
        fgaVectors[xi, yi, zi] = newVector * windScale
        countNonzero += 1
        # AND WE'RE DONE WITH INTERPOLATION
print('Interpolation complete!')

################# GRAVITY ##################

fgaVectors = np.pad(fgaVectors, tuple([(padWidth, padWidth)]) * 3 + tuple([(0, 0)]),
                    'edge')

# Next is applying a gravitational field...
# The application will go on any vector which is currently 0
print('Applying gravity...')
axisSpace = np.arange(-boxRad - padWidth * resStep,
                      boxRad + padWidth * resStep + resStep/2, resStep)
for i, x in enumerate(axisSpace, 0):
    for j, y in enumerate(axisSpace, 0):
        for k, z in enumerate(axisSpace, 0):
            if np.all(fgaVectors[i, j, k] == 0):
                if np.linalg.norm([x, y, z]) > boxRad:
                    fgaVectors[i, j, k] = [-x, -y, -z] / np.linalg.norm([x, y, z]) * gravScale
                elif np.linalg.norm([x, y, z]) < boxRad:
                    fgaVectors[i, j, k] = [x, y, z] / np.linalg.norm([x, y, z]) * gravScale
                countNonzero += 1
print('Application complete!')
print(countNonzero, 'nonzero vectors out of', DRes ** 3)

############# WRITING TO FILE ##############

# AND WE'RE DONE. Now unwrap, add in resolution and box data and write to file
# unwrap
print('Unwrapping and writing...')
print(DRes, fgaVectors.shape)
DRes += 2 * padWidth
fgaVectors = fgaVectors.reshape((DRes ** 3, 3), order='F')

fgaVectors = np.vstack(([
    [DRes, DRes, DRes],
    [-boxRad, -boxRad, -boxRad],
    [boxRad, boxRad, boxRad]
], fgaVectors))
with open('.//{}.fga'.format(varNames[0]), 'wb') as f:
    np.savetxt(f, fgaVectors, delimiter=',', newline=',\r\n', fmt='%4.7f')
print('Unwrapping and writing complete!')
end = time.clock()
print('\nFINISHED EVERYTHING IN', end - start, 'SECONDS.')
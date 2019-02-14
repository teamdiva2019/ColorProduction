import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
import sys
import time

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    if az < 0:
        az = 2 * np.pi + az
    return r, el, az

# Function to find the 4 points that make the
# box that a target lat and lon falls in.
# Returns the weights as well.
# Returns a list of tuples in
# (topLeft, topRight, bottomLeft, bottomRight) order.
def findBoxPointsAndWeights(
        targetLat: float,
        targetLon: float,
        latitudes: np.array,
        longitudes: np.array):
    upperLat = latitudes.searchsorted(targetLat)
    rightLon = longitudes.searchsorted(targetLon) % len(longitudes)
    leftLon = rightLon - 1

    # Longitude weights are normal
    # TODO: Handle the case where longitudal points don't start at 0.
    if leftLon == -1:
        diff = 2 * np.pi - longitudes[leftLon]
    else:
        diff = longitudes[rightLon] - longitudes[leftLon]
    weightLon = (targetLon - longitudes[leftLon]) / diff

    # If we're off the ends, then the four points
    # will have the same latitude, but 4 different
    # longitudes.
    if upperLat == 0 or upperLat == len(latitudes):
        # Get the opposite longitudes
        oppRightLon = (rightLon + len(longitudes) // 2) % len(longitudes)
        oppLeftLon = oppRightLon - 1
        # In the 3D world, this would result in an
        # hourglass shape, but this is intended as it does
        # not hamper the interpolations...
        # If upperlat is the length, then change it to the last
        # element.
        if upperLat == len(latitudes):
            upperLat = -1
        # The latitude weight works a tad
        # differently, because we now technically
        # go over the pole.

        # North pole:
        if upperLat == -1:
            weightLat = (targetLat - latitudes[upperLat]) / (2 * (np.pi/2 - latitudes[upperLat]))
        # South pole:
        else:
            # The subtraction turns into addition b/c negative.
            # Likewise with the addition turning into sub.
            # Net result is we have positive values.
            # NOTE: Even though it looks like it, it's not the same
            # as above.
            weightLat = (latitudes[upperLat] - targetLat) / (2 * (np.pi/2 + latitudes[upperLat]))

        return [
            (upperLat, leftLon),
            (upperLat, rightLon),
            (upperLat, oppLeftLon),
            (upperLat, oppRightLon)
        ], weightLat, weightLon
    # Otherwise, calculate the lower latitude
    # as normal
    lowerLat = upperLat - 1
    # Latitude weight is normal
    weightLat = (targetLat - latitudes[upperLat]) / (latitudes[upperLat] - latitudes[lowerLat])
    return [
        (upperLat, leftLon),
        (upperLat, rightLon),
        (lowerLat, leftLon),
        (lowerLat, rightLon)
    ], weightLat, weightLon


# Function to find the closest 2 points to another point
def find2Points(p, pointSpace, vectSpace):
    ds = np.linalg.norm(pointSpace - p, axis=1)
    ls = ds.argsort()[:2]
    return pointSpace[ls], vectSpace[ls], ds[ls]

def writeFGAFile(data, radius, resStep, padWidth=1):
    # Set the size of the box and
    # the resolution step...user inputs.
    # Also calculate diameter in res units
    DRes = int(2 * radius / resStep + 1)
    print('DRes:', DRes)
    # Set scaling values
    windScale = 30
    gravScale = 30

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

    #### For horizontal vector field ###
    # numLatPoints = 10
    # numLonPoints = 10
    #
    # directions = np.zeros(numLatPoints * numLonPoints).reshape((numLatPoints, numLonPoints))
    # u = np.cos(directions)
    # v = np.sin(directions)
    # plt.quiver(u, v)
    # plt.show()
    #
    # latPoints = np.arange(-90 + 180 / (numLatPoints + 1),
    #                       90 - 180 / (numLatPoints + 1) + 0.01,
    #                       180 / (numLatPoints + 1)) * np.pi / 180
    # lonPoints = np.arange(0, 360 - 360 / numLonPoints, 360 / numLonPoints)

    u, v = allVariables[varNames[0]][0], allVariables[varNames[1]][0]
    # print(u[:2], v[:2])
    # plt.quiver(lonPoints * 180 / np.pi, latPoints * 180 / np.pi, u, v)
    # plt.savefig('plotDownSample.png', dpi=300)
    # plt.show()



    ################# TRANSFORM TO 3D VECTORS ##################
    print('Transforming to 3D vectors...')
    vects3D = np.zeros((numLatPoints, numLonPoints, 3))
    vects3D[:, :, 0] = u
    vects3D[:, :, 1] = v
    # print('Initial 3D vectors:', vects3D)
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
                    [np.cos(np.pi / 2 - lat), 0, np.sin(np.pi / 2 - lat)],
                    [0, 1, 0],
                    [-np.sin(np.pi / 2 - lat), 0, np.cos(np.pi / 2 - lat)]
                ]
            )
            # Before we rotate, we need to convert to a "top-down" view,
            # so the +x-axis will point down and the +y-axis will point right.
            # (x,y,0) ==> (-y,x,0)
            np.dot(rotMatrix, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), out=rotMatrix)
            # Apply the entire rotation matrix
            # print(vects3D[i,j], '\t\t==> ', end='')
            vects3D[i, j] = np.dot(rotMatrix, vects3D[i, j])
            # print(vects3D[i,j], '\t\t==> ', end='')

            ### Calculate the 2d point. Basically spherical coordinates
            exactVectLocs[i, j] = radius * np.array([
                np.sin(np.pi / 2 - lat) * np.cos(lon),
                np.sin(np.pi / 2 - lat) * np.sin(lon),
                np.cos(np.pi / 2 - lat)
            ])
            # print('at', exactVectLocs[i, j], '({:.2f}, {:.2f})'.format(lat * 180 / np.pi, lon * 180 / np.pi))
    print('Transform complete!')
    # print('3D Transformation after:', vects3D)
    # print()
    # print('Exact vector locations:', exactVectLocs)
    # sys.exit(1)

    ################# GRAVITY ##################

    # # Next is applying a gravitational field...
    # # The application will go on any vector which is currently 0
    print('Applying gravity...')
    padRadius = radius + padWidth * resStep
    fgaVectors = np.transpose(np.mgrid[-padRadius : padRadius + resStep/2 : resStep,
                 -padRadius : padRadius + resStep/2 : resStep,
                 -padRadius: padRadius + resStep/2 : resStep], (1,2,3,0))
    # Normalize and flip vectors that
    # are > radius away...
    norms = np.linalg.norm(fgaVectors, axis=-1)
    fgaVectors /= norms[:, :, :, None]
    fgaVectors[norms > radius] = -1 * fgaVectors[norms > radius]
    fgaVectors *= gravScale
    print(fgaVectors.shape)
    print('Application complete!')

    ################# INTERPOLATION ######################

    print('Interpolating vectors...')
    # So here is the plan: We will first make
    # a 3D (really 4D) array where each element
    # is the actual [x, y, z] coordinate across
    # the whole box.
    # Then, we go through each point, and see if it's
    # "close enough" to the surface to the sphere
    # (not inside).
    # If it is, then we calculate that point's
    # latitude and longitude values, using a conversion
    # to spherical coordinates.
    # We then check which 2 latitudes and longitudes we
    # have actual data on. Each of our [x,y,z]
    # points will fall in a rectangle where the corners
    # have actual wind data.
    # At this point, we can interpolate horizontally
    # (using the 2 longitudal sides), and interpolate
    # vertically (using the 2 latitude sides) across
    # both angle and magnitude. Do this for every
    # point and we have finished our interpolation.

    ### LET'S GET STARTED! ###

    # First we need a way to iterate through all possible
    # point values, along with their respective indices,
    # so we can assign the proper values in fgaVectors.
    # Because this a square box, this is made easy
    # through itertools.product(). We make one each for
    # the point values and index values. We then iterate
    # through them at the same time using zip().

    # From our padding, value space goes from radius
    # to radius. But our index space now starts at
    # the padWidth, and goes the length of the value space.
    valueSpace = np.arange(-radius, radius + 1e-10, resStep)
    indexSpace = np.arange(padWidth, padWidth + len(valueSpace))
    valueProduct = product(valueSpace, valueSpace, valueSpace)
    indexProduct = product(indexSpace, indexSpace, indexSpace)
    for point, indices in zip(valueProduct, indexProduct):
        x, y, z = point
        xi, yi, zi = indices
        distFromOrigin = np.linalg.norm([x,y,z])
        # If the distance from the point to
        # the origin is "close enough", then we
        # interpolate on this point. Here, "close enough"
        # means above the sphere and less than resStep away.
        if 0 <= distFromOrigin - radius < resStep:
            # print('({}, {}, {}) ==> '.format(x, y, z), end='')
            # print('[{}, {}, {}] ==> '.format(xi, yi, zi), end='')

            # Find the spherical coordinates of this point.
            # The method returns the latitude and longitude
            # in the correct ranges.
            rho, lat, lon = cart2sph(x, y, z)
            # print('({:.2f}, {:.2f}, {:.2f}) ==> '.format(rho, lat * 180 / np.pi, lon * 180 / np.pi), end='')

            # Use numpy's fancy searchsorted function to find
            # the closest locations for latitude and longitude.
            # searchsorted returns a right associated index.
            # The reason for the mod is that if the value is
            # off the deep end, then searchsorted returns
            # the length of the array, which is invalid index.
            # So mod the length the array to turn it 0, and
            # the left index (which should wrap around), works as
            # intended.

            boxLocs, weightLat, weightLon = findBoxPointsAndWeights(lat, lon, latPoints, lonPoints)

            # Point 4 is assumed to be diagonally opposite Point 1,
            # and Point 2 is assumed to be horizontally opposite Point 1.
            # First we need to find the two vectors directly
            # above and below (left and right also works) our
            # target location.
            interAbove = (1-weightLon) * vects3D[boxLocs[0]] + weightLon * vects3D[boxLocs[1]]
            interBelow = (1-weightLon) * vects3D[boxLocs[2]] + weightLon * vects3D[boxLocs[3]]
            # Now we interpolate vertically
            # between these two points

            interedVec = (1-weightLat) * interBelow + weightLat * interAbove

            # print('({:.2f}, {:.2f}, {:.2f})'.format(interedVec[0], interedVec[1], interedVec[2]))

            # We're done with the interpolation with this vector,
            # so now using the index values all the way above, assign
            # to fgaVectors...scaling as necessary
            fgaVectors[xi, yi, zi] = interedVec * windScale

    ############# WRITING TO FILE ##############

    # AND WE'RE DONE. Now unwrap, add in resolution and box data and write to file
    # unwrap
    # print(fgaVectors)
    print('Unwrapping and writing...')
    print(DRes, '==>', DRes + 2 * padWidth)
    DRes += 2 * padWidth
    fgaVectors = fgaVectors.reshape((DRes ** 3, 3), order='F')

    fgaVectors = np.vstack(([
                                [DRes, DRes, DRes],
                                [-padRadius, -padRadius, -padRadius],
                                [padRadius, padRadius, padRadius]
                            ], fgaVectors))
    with open('.//{}.fga'.format(varNames[0]), 'wb') as f:
        np.savetxt(f, fgaVectors, delimiter=',', newline=',\r\n', fmt='%4.7f')
    # with open('.//noGravity.fga', 'wb') as f:
    #     np.savetxt(f, fgaVectors, delimiter=',', newline=',\r\n', fmt='%4.7f')
    print('Unwrapping and writing complete!')

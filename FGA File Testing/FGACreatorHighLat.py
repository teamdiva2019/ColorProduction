import time
import numpy as np
import matplotlib.pyplot as plt
import os

start = time.clock()

# Get script directory
scriptDir = os.path.dirname(os.path.realpath(__file__))
latPoints = 5
lonPoints = 10
# Make a 5 x 10 vector grid
directions = np.zeros(latPoints * lonPoints).reshape((latPoints, lonPoints))
# # directions[0, 0] = np.pi / 4
# # directions[1, 0] = np.pi / 2
u = np.cos(directions)
v = np.sin(directions)
plt.quiver(u, v)
plt.show()

latBounds = np.array([-90 + 180/(latPoints+1),
                      90 - 180/(latPoints+1), 180/(latPoints+1)]) * np.pi / 180
lonBounds = np.array([0, 360 - 360/lonPoints, 360/lonPoints]) * np.pi / 180
print('Latitude Bounds:', latBounds)
print('Longitude Bounds:', lonBounds)

# Set constants
windScale = 500  # how much to scale the vectors by
gravScale = 100
actRad = 200  # actual radius
minBox = [(-1) * actRad] * 3
maxBox = [actRad] * 3
# distance between top vectors
distBet = 1
rTop = distBet / lonBounds[2]
# Diameter of sphere in resolution units i.e. not the actual diameter
D = int(2 * rTop / np.sin(latBounds[2]) + 1)
resStep = (2 * actRad) / (D - 1)
print(D, resStep)

vects3D = np.zeros(3*latPoints*lonPoints).reshape((latPoints, lonPoints, 3))
vects3D[:, :, 0] = u
vects3D[:, :, 1] = v
# print(vects3D)

for latInd in range(latPoints):
    for lonInd in range(lonPoints):
        lat = latBounds[0] + latInd * latBounds[2]
        lon = lonBounds[0] + lonInd * lonBounds[2]
        # Find the rotation matrix
        rotMatrix = np.dot(
            np.array([
                [np.cos(lon), -np.sin(lon), 0],
                [np.sin(lon), np.cos(lon), 0],
                [0, 0, 1]
            ]),
            # Subtract from pi/2 to get the rotation correctly
            np.array([
                [np.cos(np.pi/2 - lat), 0, np.sin(np.pi/2 - lat)],
                [0, 1, 0],
                [-np.sin(np.pi/2 - lat), 0, np.cos(np.pi/2 - lat)]
            ])
        )
        # Convert to a "top-down view" of the vectors. This causes
        # (x, y, 0) to go to (-y, x, 0)
        np.dot(rotMatrix, np.array([[0,-1,0],[1,0,0],[0,0,1]]), out=rotMatrix)
        # Rotate the vector at the location
        print(vects3D[latInd, lonInd], 'at',
              '({}, {})'.format(lat * 180/np.pi, lon * 180/np.pi), end=' -> ')
        vects3D[latInd, lonInd] = windScale * np.dot(rotMatrix, vects3D[latInd, lonInd])
        print(vects3D[latInd, lonInd])

print()

#################### APPLYING THE VECTORS #####################

fgaVectors = np.zeros((D, D, D, 3))
for latInd in range(latPoints):
    for lonInd in range(lonPoints):
        lat = latBounds[0] + latInd * latBounds[2]
        lon = lonBounds[0] + lonInd * lonBounds[2]
        x, y, z = (
            actRad * np.sin(np.pi/2 - lat) * np.cos(lon),
            actRad * np.sin(np.pi/2 - lat) * np.sin(lon),
            actRad * np.cos(np.pi/2 - lat)
        )
        # Find the index
        xi, yi, zi = (
            int(np.round((x + actRad) / resStep)),
            int(np.round((y + actRad) / resStep)),
            int(np.round((z + actRad) / resStep))
        )
        # print('({}, {})'.format(lat * 180/np.pi, lon * 180/np.pi), '->',
        #       '({}, {}, {})'.format(x, y, z), '->',
        #       '[{}, {}, {}]'.format(xi, yi, zi),
        #       '-> Distance:', np.round(np.linalg.norm([x, y, z])))
        # print(vects3D[latInd, lonInd])
        # Set it equal to it in our fga array
        fgaVectors[xi, yi, zi] = vects3D[latInd, lonInd]

        ############# Setting adjacent ones to the same as well
        if xi != 0 and yi != 0 and zi != 0 and xi < D - 1 and yi < D - 1 and zi < D - 1:
            for xii in range(xi-1, xi+2):
                for yii in range(yi-1, yi+2):
                    for zii in range(zi-1, zi+2):
                        fgaVectors[xii, yii, zii] = vects3D[latInd, lonInd]

        # print(fgaVectors[xi, yi, zi])
print([D, D, D])
print(minBox)
print(maxBox)

################# PUTTING IN THE GRAVITY ####################

# Now to create an inward gravitational field outside the
# sphere so any particles get pulled in.
# Just a standard through of each point
# count = 0
for xi in range(fgaVectors.shape[0]):
    for yi in range(fgaVectors.shape[1]):
        for zi in range(fgaVectors.shape[2]):
            x, y, z = (
                -actRad + xi * resStep,
                -actRad + yi * resStep,
                -actRad + zi * resStep
            )

            if np.linalg.norm([x, y, z]) > 100:
                fgaVectors[xi, yi, zi] = [-x, -y, -z] / np.linalg.norm([x, y, z]) * gravScale
            # count += 1


# unwrap
fgaVectors = fgaVectors.reshape((D ** 3, 3), order='F')
# print(fgaVectors[0])
print(fgaVectors.shape)

fgaVectors = np.vstack(([[D, D, D], minBox, maxBox], fgaVectors))
with open('.//justGravity.fga', 'wb') as f:
    np.savetxt(f, fgaVectors, delimiter=',', newline=',\r\n', fmt='%4.7f')

end = time.clock()
print(end - start, 'seconds.')
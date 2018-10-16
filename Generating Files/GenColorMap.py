import numpy as np
import sys
import time

# Takes one color and converts to Hunter
def RGBToHunter(color):
    # Special case
    if np.array_equal(color, [0, 0, 0]):
        return np.array([0, 0, 0.])
    M = np.array([
        [0.412383, 0.357585, 0.18048],
        [0.212635, 0.71517, 0.072192],
        [0.01933, 0.119195, 0.950528]
    ])
    Xn, Yn, Zn = 95.047, 100, 108.883
    Ka, Kb = (175/198.04)*(Xn + Yn), (70/218.11)*(Yn + Zn)

    color /= 255
    bools = color > 0.04045
    color[bools] = ((color[bools] + 0.055) / 1.055) ** 2.4
    color[~bools] /= 12.92
    color = M.dot(color * 100)
    # Now color has X, Y, Z
    saved = color[1] / Yn
    L = 100 * np.sqrt(saved)
    a = Ka * (color[0] / Xn - saved) / np.sqrt(saved)
    b = Kb * (saved - color[2] / Zn) / np.sqrt(saved)

    return np.array([L, a, b])

# Takes list of color in lab and converts to
# RGB in place
def hunterToRGB(colors):
    Minv = np.array([
        [3.24103, -1.53741, -0.49862],
        [-0.969242, 1.87596, 0.041555],
        [0.055632, -0.203979, 1.05698]
    ])
    Xn, Yn, Zn = 95.047, 100, 108.883
    Ka, Kb = (175 / 198.04) * (Xn + Yn), (70 / 218.11) * (Yn + Zn)
    # L, a, b are down the columns, make them rows
    colors = colors.T
    saved = colors[0] / 100
    # Calculate X, Y, Z
    colors[0] = saved * (colors[1] / Ka + saved) * Xn
    colors[1] = saved ** 2 * Yn
    colors[2] = saved * (saved - colors[2] / Kb) * Zn
    # Multiply by the matrix
    # colors = Minv.dot(colors / 100)
    colors = Minv.dot(colors / 100)
    # Apply corrections

    bools = colors > 0.0031308
    colors[bools] = 1.055 * (colors[bools] ** (1/2.4)) - 0.055
    colors[~bools] *= 12.92
    # Clip extremes and return
    return np.clip(colors, 0, 1, out=colors).T

def runScript(argv):
    start = time.clock()

    # Read the input, first check for correct number
    if (len(argv) - 1) % 3 != 0:
        print('Invalid colors!')
        exit(1)
    # Parse the colors and read the map name
    mapName = argv[0]
    stops = np.array(list(map(float, argv[1:]))).reshape((len(argv[1:]) // 3, 3))
    # Convert the gradient stops to colors
    for i in range(len(stops)):
        stops[i] = RGBToHunter(stops[i])

    totalCols = 256
    colsPerStop = int((totalCols - len(stops)) / (len(stops) - 1) + 1)
    gradient = np.zeros((colsPerStop * (len(stops) - 1) + 1, 3))
    for i in range(len(stops) - 1):
        redRange = np.linspace(stops[i, 0], stops[i+1, 0],
                               colsPerStop, endpoint=False)[np.newaxis,:]
        greenRange = np.linspace(stops[i, 1], stops[i+1, 1],
                                 colsPerStop, endpoint=False)[np.newaxis,:]
        blueRange = np.linspace(stops[i, 2], stops[i+1, 2],
                                colsPerStop, endpoint=False)[np.newaxis,:]
        # Concatenate the ranges to get a range of points that
        # move directly to the endpoint
        fullRange = np.concatenate((redRange, greenRange, blueRange), axis=0).T
        # And add to the upper gradient array
        gradient[i * colsPerStop:(i+1) * colsPerStop] = fullRange
    # Add the final color
    gradient[-1] = stops[-1]
    # Convert to RGB
    finalMap = hunterToRGB(gradient)
    # Save the final map
    np.savetxt('.//ColorMaps//' + mapName + '.txt', finalMap, fmt='%1.10f')

    end = time.clock()
    print(end - start, 'seconds.')

if __name__ == '__main__':
    runScript(sys.argv[1:])
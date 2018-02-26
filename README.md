# Color Production
This file provides the framework to map data into colors. This is mainly used in the Unreal Engine in order to
streamline the process of visualizing the data. Through this method, Unreal itself will essentially have no
direct knowledge of the data, just the colors. There are two files: GenColorMap.py and GenColorsFromData.py

## GenColorMap.py
This generates a perceptually uniform map given a name and a list of color stops. The number of stops
can be as long as possible and the program will generate a map in 256 steps, putting the stops
in equal distances to form a gradient. The generated map will be saved to a text file under the name given
to the ColorMaps folder. This is then used by GenColorsFromData.py in order to map data according to a
specific color map. See this file description below.

## GenColorsFromData.py
This program is the main, so to speak. A netcdf file will be given, along with the name of the color map
to map to. If the color map does not exist in the ColorMaps folder, then it is assumed that the required colors
will be given.
From the file, metadata.txt will be generated, which contains the bare minimum description of the variables
and the dimensions necessary for Unreal to potentially display and label the final visualization.
If only a single variable is present, then a single binary file is produced in ColorsFromData which contains the
mapping. The RGB values are in 0-255, to save space.
If there is more than one variable, then it is assumed that the variable is vectorized. In addition to the colormap
file, a directions file will be produced and placed in the Directions folder. This contains the directions of
each data point. Note that only 2-dimensional vectors are supported.
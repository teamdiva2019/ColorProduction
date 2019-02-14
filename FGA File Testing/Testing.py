from netCDF4 import Dataset
import numpy as np
import sys

# Read file path
file_path = sys.argv[1]

data = Dataset(file_path)
allVariables = data.variables
print(allVariables)

# Sometimes we have time_bnds, lat_bnds, etc.
# Keep anything that doesn't have 'bnds'
varNames = list(filter(lambda x: 'bnds' not in x, list(allVariables.keys())))
# Remove the dimensions
varNames = list(filter(lambda x: x not in data.dimensions, varNames))

for var in varNames:
    data = allVariables[var][0]
    print(var, data)
print(allVariables['latitude'][:])
print(allVariables['longitude'][:])
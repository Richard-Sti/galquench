from copy import deepcopy
import numpy


def subhalos_to_array(subhalos, legend):
    # Will be popping so deepcopy
    legend = deepcopy(legend)
    # Extrac the data
    out = {}
    for leg in legend:
        param, col = leg
        data = subhalos[param]
        # If a column was specified take only that
        if col is not None:
            data = data[:, col]
            param += "_{}".format(col)
        out.update({param: data})

    # Convert to a structured array
    dtype = {'names': [key for key in out.keys()],
             'formats': [d.dtype for d in out.values()]}
    arr = numpy.zeros(data.size, dtype=dtype)
    for key, val in out.items():
        arr[key] = val
    return arr

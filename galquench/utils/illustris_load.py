from copy import deepcopy
import numpy
import h5py


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


def read_supplementary(file, subfindID, keys=None, skip_keys=None, snapshot_number=None):
    data = h5py.File(file, "r")
    # Get the snapshot
    if snapshot_number is not None:
        snapshot = "Snapshot_{}".format(snapshot_number)
        if snapshot not in data.keys():
            raise ValueError("Invalid snapshot number `{}`".format(snapshot_number))
        # Get just this snapshot
        data = data[snapshot]
    # Check if a specific key(s) given
    if keys is None:
        keys = list(data.keys())
        keys.remove(subfindID)
    elif isinstance(keys, str):
        keys = [keys]
    else:
        for key in keys:
            if key not in data.keys():
                raise ValueError("Invalid key `{}`.".format(key))

    # If any skip keys, remove them
    if skip_keys is not None:
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        for skip_key in skip_keys:
            if skip_key in keys:
                keys.remove(skip_key)

    # Put into a dictionary and return
    out = {"subfindID": numpy.asarray(data[subfindID]).astype(int)}
    for key in keys:
        out.update({key: numpy.asarray(data[key])})
    return out

from copy import deepcopy
import numpy
import h5py


def read_supplementary(file, subfindID, keys=None, skip_keys=None,
                       snapshot_number=None):
    """
    Read a supplementary TNG catalogue, assumed in `hdf5` format and return
    specific keys along with the corresponding subhalo IDs.

    Arguments
    ---------
    file : str or :py:class:`h5py._hl.files.File`
        File (path).
    keys : (list of) str, optional
        Keys to be returned. By default `None`, all keys are returned.
    skip_keys : (list of) str, optional
        Keys to be skipped if e.g. all keys are to be returned. By default
        `None`, no keys are skipped.
    snapshot_numer : int, optional
        The snapshot number. If `None` assumed that the specified file is
        already for a given snapshot.

    Returns
    -------
    out : dict
        Dictionary with the requested data.
    """
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
        d = data[key]
        if isinstance(d, h5py._hl.dataset.Dataset):
            d = numpy.asarray(d)
        out.update({key: d})
    return out

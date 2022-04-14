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
    # Check that this is not a snapshot file
    if any("Snapshot_" in key for key in data.keys()):
        raise ValueError("This appears to be a file with snapshots. "
                         "Specify `snapshot_number`.")
    # Get keys
    if keys is None:
        keys = list(data.keys())
    elif isinstance(keys, str):
        keys = [keys]
    elif isinstance(keys, list) and all(isinstance(key, str) for key in keys):
        pass
    else:
        raise ValueError("`keys` must be either a string or a list of strings.")

    #Check that we have a good subfind key
    if subfindID not in data.keys():
        suggestions = [key for key in data.keys() if "id" in key.lower()]
        raise ValueError("subfindID of `{}` is invalid. Possibly one of `{}`."
                         .format(subfindID, suggestions))
    # Remove subfind ID from the list of keys
    if subfindID in keys:
        keys.remove(subfindID)

    # Check that all keys are in the data file
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


def unpack_catalog_columns(catalog, column_mapping):
    fields = [
        key for key in catalog.keys() if key not in ["count", "subfindID"]]
    for field in fields:
        if not isinstance(catalog[field], numpy.ndarray):
            raise TypeError("Field `{}` is not an array.".format(field))

        if catalog[field].ndim == 1:
            continue

        if field not in column_mapping.keys():
            raise RuntimeError("Column information for field `{}` required."
                               .format(field))

        columns = column_mapping[field]
        if not isinstance(columns, (list, tuple)):
            cols = [cols]

        data = catalog.pop(field)
        for column in columns:
            catalog.update({field + "_{}".format(column): data[:, column]})
    return catalog


def match_subfind(catalogs, N):
    for cat in catalogs:
        for key in cat.keys():
            if key == "subfindID":
                continue
            data = cat[key]
            full = numpy.full(N, fill_value=numpy.nan, dtype=data.dtype)
            full[cat["subfindID"]] = data
            cat.update({key: full})

    for cat in catalogs:
        cat.pop("subfindID")
    return catalogs


def merge_subhalos_with_supplementary(subhalos, supplementary_cats):
    N = subhalos.pop("count")
    dtype = {"names": [], "formats": []}
    for cat in [subhalos] + supplementary_cats:
        for key, value in cat.items():
            dtype["names"].append(key)
            dtype["formats"].append(value.dtype.type)

    arr = numpy.full(N, numpy.nan, dtype=dtype)

    for cat in [subhalos] + supplementary_cats:
        for key, value in cat.items():
            arr[key] = value
    return arr
from copy import deepcopy
from warnings import warn
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
    """
    Match data from supplementary catalogs by the subfind IDs by creating
    a 1-dimensional array with data at positions specified by `subfindID` and
    NaNs elsewhere.

    Arguments
    ---------
    catalogs : list of dictionaries
        Supplementary catalogs from py:func:`read_supplementary`.
    N : int
        Count of subfind halos at the specified snapshot.

    Returns
    -------
    catalogs : list of dictionaries
        Supplementetary catalogs whose array ordering matched the subfind table.
    """
    # Optionally convert to a list if only one catalog passed in
    if not isinstance(catalogs, list):
        catalogs = list(catalogs)
    # Check everything is a dictionary
    if not all(isinstance(catalog, dict) for catalog in catalogs):
        raise TypeError("Entries of `catalogs` must be dictionaries.")
    for catalog in catalogs:
        for key in catalog.keys():
            # Ignore subfind ID
            if key == "subfindID":
                continue
            # Convert to an array matchign the sublos with the subfind IDs
            # Put NaNs where no entry in this supplementary catalog
            data = catalog[key]
            full = numpy.full(N, fill_value=numpy.nan, dtype=data.dtype)
            full[catalog["subfindID"]] = data
            catalog.update({key: full})
    # Now that we matched by subfindID drop it
    for catalog in catalogs:
        catalog.pop("subfindID")
    return catalogs


def merge_subhalos_with_supplementary(subhalos, supplementary_catalogs):
    """
    Merge the `subhalos` dictionary with `supplementary_cats` dictionaries
    which have been rearranged by their subfind IDs via
    py:func:`match_subfind`.

    Arguments
    ---------
    subhalos : dict
        Subhalos dictionary from py:func:`loadSubhalos`.
    supplementary_catalogs : list of dictionaries
        Supplementary catalogs matched by their subfind IDs.

    Returns
    -------
    arr : structured array
        Single structured array with the merged data.
    """
    if not isinstance(subhalos, dict):
        raise TypeError("`subhalos` must be a dictionary.")
    if not all(isinstance(catalog, dict) for catalog in supplementary_catalogs):
        raise TypeError("`supplementray_cats` must be a list of dictionaries.")

    N = subhalos.pop("count", None)
    if N is None:
        raise ValueError("Subhaloes must contain key `count`.")
    # Get the dtypes
    dtype = {"names": [], "formats": []}
    for catalog in [subhalos] + supplementary_catalogs:
        for key, value in catalog.items():
            dtype["names"].append(key)
            dtype["formats"].append(value.dtype.type)
    # Initialise the array and fill it
    arr = numpy.full(N, numpy.nan, dtype=dtype)
    for catalog in [subhalos] + supplementary_catalogs:
        for key, value in catalog.items():
            arr[key] = value
    return arr


def apply_multiplicative_units(array, units):
    """
    Apply multiplicative units, if such unit is specified in `array`.

    Arguments
    ---------
    array : structured array
        Structured array with data.
    units : dict
        Dictionary where keys are the parameters (of `array`) are to be
        multiplied by their value.

    """
    for name in array.dtype.names:
        for unit, factor in units.items():
            if unit.lower() in name.lower():
                msg = "Multiplying `{}` by {}.".format(name, factor)
                warn(msg, RuntimeWarning)
                array[name] *= factor
    return array


def apply_selection(array, selection, only_finite=False):
    """
    Apply lower and upper limit selection to array.

    Arguments
    ---------
    array : structured array
        Structured array with data to be selected.
    selection : dict
        Keys must be parameters and values the upper and lower limit.
    only_finite : bool, optional
        Whether to remove all samples that contain a NaN in any parameter.
        By default `False`.

    out : structured array
        Data array with selection applied.
    """
    masks = [None] * len(selection)
    for i, (par, lims) in enumerate(selection.items()):
        # Checks lims specified in a good format
        if not isinstance(lims, (tuple, list)) or len(lims) != 2:
            raise TypeError("`lims` of parameter `{}` must be a list or "
                            "tuple of length 2".format(par))
        lower, upper = lims
        masks[i] = (lower < array[par]) & (array[par] < upper)

    if only_finite:
        for param in array.dtype.names:
            masks.append(numpy.isfinite(array[param]))

    final_mask = masks[0]
    if len(masks) > 1:
        for mask in masks[1:]:
            final_mask &= mask
    Nrem = numpy.sum(~final_mask)
    N = final_mask.size
    warn("Removing {} ({:.2f}%) objects.".format(Nrem, Nrem / N * 100))

    return array[final_mask]

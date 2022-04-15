import numpy

from os.path import isfile
import sys
sys.path.append("../")
from galquench.utils import (
    loadSubhalos, read_supplementary, unpack_catalog_columns, match_subfind,
    merge_subhalos_with_supplementary, apply_multiplicative_units,
    apply_selection)


output_path = "/Users/richard/Projects/galquench/data/test.npy"
folder = "/Users/richard/Projects/galquench/data/tng100/"
only_finite = False

################################################################################
#                       Subhalo file settings                                  #
################################################################################


subhalos_file = {"basePath": folder + "groupcats",
                 "snapNum": 99,
                 "fields": None}
subhalos_field = [
    "SubhaloMassType", "SubhaloStellarPhotometricsRad"
    ]
subhalos_file.update({"fields": subhalos_field})


################################################################################
#                       Supplementary file settings                            #
################################################################################


supplementary_files = [
    {"file" : folder + "star_formation_rates.hdf5",
     "subfindID": "SubfindID",
     "snapshot_number": 99,
     "keys": ["SFR_MsunPerYrs_in_all_1000Myrs"]},
    #
    {"file" : folder + "hih2_galaxy_099.hdf5",
     "subfindID": "id_subhalo",
     "keys": ["m_neutral_H"]}
    ]


################################################################################
#                       Units, columns and selection                           #
################################################################################


multiplicative_units = {"SubhaloMass": 1e10}
column_mapping = {"SubhaloMassType": (1, 4)}
selection = {"SubhaloMassType_1" : (0, 1e12),
             "SubhaloMassType_4" : (0, 1e12)}


################################################################################
#                             Load everything                                  #
################################################################################


subhalos = loadSubhalos(**subhalos_file)
supplementary_cats = [None] * len(supplementary_files)
for i, f in enumerate(supplementary_files):
    supplementary_cats[i] = read_supplementary(**f)


################################################################################
#              If any further work with supplementary catalogs, do here        #
################################################################################


################################################################################
#                       Unpack, match, units and select                        #
################################################################################


for catalog in [subhalos] + supplementary_cats:
    unpack_catalog_columns(catalog, column_mapping)

match_subfind(supplementary_cats, subhalos["count"])
arr = merge_subhalos_with_supplementary(subhalos, supplementary_cats)
apply_multiplicative_units(arr, multiplicative_units)

out = apply_selection(arr, selection, only_finite=only_finite)

ans = 'Y'
if isfile(output_path):
    inps = ['Y', 'n']
    while True:
        ans = input("File `{}` exists. Overwrite? [Y, n] ".format(output_path))
        if ans in inps:
            break
        print("Invalid input `{}`. Must be one of `{}`".format(ans, inps))

if ans == 'Y':
    numpy.save(output_path, out)
    print("Job completed. Output saved to `{}`.".format(output_path))
else:
    print("Job completed but not saved.")

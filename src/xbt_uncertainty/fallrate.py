"""Fall-rate equations z = a t - b t^2, with z in metres and t in seconds.

Coefficients are from the original papers:

Hanawa, K., Rual, P., Bailey, R., Sy, A. and Szabados, M. (1994). Calculation
of new depth equations for expendable bathythermographs using a
temperature-error-free method (application to Sippican/TSK T-7, T-6 and T-4
XBTs). UNESCO Technical Papers in Marine Science 67 (IOC Technical Series 42).
    Manufacturer equation for T-7, T-6 and T-4: Table 2.
    New equation for T-7, T-6 and T-4: Eq. 9.
"""

import numpy as np

MANUFACTURER_T4_T6_T7 = (6.472, 2.16e-3)
HANAWA_1994 = (6.691, 2.25e-3)

# Correction applied to each predicted class when its stored depths came from
# the manufacturer equation. A class is in the Hanawa group if WOD flags
# labelled casts of that model as "needs Hanawa et al.; 1994 applied (XBT)";
# models that WOD never flags this way get no depth correction here.
# scripts/correct_unknown_profiles.py checks this table against the data.
HANAWA_GROUP = {"T4", "T6", "T7", "DEEP BLUE", "XBT-7"}
NO_CORRECTION_GROUP = {"T5", "T10", "T11", "FAST DEEP", "OTHER"}


def depth_from_time(t, a, b):
    return a * t - b * t**2


def time_from_depth(z, a, b):
    """Inverse of depth_from_time on the descending branch."""
    z = np.asarray(z, dtype=float)
    return (a - np.sqrt(a**2 - 4 * b * z)) / (2 * b)


def hanawa_from_manufacturer(z):
    """Depth under Hanawa et al. (1994) for a depth stored with the manufacturer equation."""
    t = time_from_depth(z, *MANUFACTURER_T4_T6_T7)
    return depth_from_time(t, *HANAWA_1994)


def hanawa_eq12(z):
    """Hanawa et al. (1994) Eq. 12, their closed-form conversion. Used only in tests."""
    z = np.asarray(z, dtype=float)
    return 1.0417 * z - 75.906 * (1 - np.sqrt(1 - 2.063e-4 * z))

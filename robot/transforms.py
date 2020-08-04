import numpy as np
from robopy.robot.util import ishomog

def t2r(tmat):
    """
    Convert homogeneous transform to a rotation matrix

    :param tmat: homogeneous transform
    :return: rotation matrix

    T2R(tmat) is the orthonormal rotation matrix component of homogeneous
    transformation matrix tmat.  Works for T 3x3 or 4x4
    if tmat is 3x3 then return is 2x2, or
    if tmat is 4x4 then return is 3x3.
    """
    if ishomog(tmat):
        return tmat[0:-1, 0:-1]

    return False
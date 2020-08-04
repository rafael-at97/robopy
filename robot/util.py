import numpy as np

def ishomog(tmat):
    """
    Test if input is homogeneous transformation matrix.
    
    :param matrix: Matrix for verification

    Validity of rotational part is not checked
    """
    if not isinstance(tmat, np.ndarray):
        raise AttributeError
    if not isinstance(tmat[0], np.ndarray):
        raise AttributeError
    if tmat.shape not in [(4, 4), (3, 3)]:
        raise ValueError("Matrix must be square 3x3 or 4x4")

    return True

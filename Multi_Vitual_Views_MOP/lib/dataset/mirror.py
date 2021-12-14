import numpy as np
from os.path import join
import os
import cv2

FLIP_BODY25 = [0,1,5,6,7,2,3,4,8,12,13,14,9,10,11,16,15,18,17,22,23,24,19,20,21]

def flipPoint2D(point):
    if point.shape[-2] == 25:
        return point[..., FLIP_BODY25, :]
    elif point.shape[-2] == 15:
        return point[..., FLIP_BODY25[:15], :]

# Permutation of SMPL pose parameters when flipping the shape
_PERMUTATION = {
    'smpl': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22],
    'smplh': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 24, 25, 23, 24],
    'smplx': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 24, 25, 23, 24, 26, 28, 27],
    'smplhfull': [
        0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, # body
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
    ],
    'smplxfull': [
        0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, # body
        22, 24, 23,  # jaw, left eye, right eye
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, # right hand
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, # left hand
    ]
}
PERMUTATION = {}
for key in _PERMUTATION.keys():
    res = []
    for i in _PERMUTATION[key]:
        res.extend([3*i + j for j in range(3)])
    PERMUTATION[max(res)+1] = res

def flipSMPLPoses(pose):
    """Flip pose. 
    const input: (N, 72) -> (N, 72)
    The flipping is based on SMPL parameters.
    """
    N_params = pose.shape[0]
    pose = pose[PERMUTATION[N_params]]

    if N_params in [72, 156, 165]:
        pose[1::3] = -pose[1::3]
        pose[2::3] = -pose[2::3]
    elif N_params in [78, 87]:
        pose[1:66:3] = -pose[1:66:3]
        pose[2:66:3] = -pose[2:66:3]
    else:
        import ipdb; ipdb.set_trace()
    # we also negate the second and the third dimension of the axis-angle
    return pose
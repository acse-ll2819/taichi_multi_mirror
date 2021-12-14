import numpy as np
import math


def simple_assign(keypoints3d):
    """
    find the id of the real view
    """
    root_coordinates = []
    nViews = keypoints3d.shape[0]
    for i in range (nViews):
        root_coordinates.append(keypoints3d[i][0])
    distances = []
    for i in range(nViews):
        distance = math.sqrt(root_coordinates[i][0]**2+root_coordinates[i][1]**2+root_coordinates[i][2]**2)
        distances.append(distance)

    distances = np.array(distances)
    id_min = np.argmin(distances)
    return id_min
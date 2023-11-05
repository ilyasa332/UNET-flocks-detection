from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt, patches
from scipy.spatial.distance import cdist


def set_cover(universe, subsets):
    # Create a copy of the universe as a set
    universe_set = set(universe)

    # Initialize an empty list to store the selected subsets
    selected_subsets = []

    while universe_set:
        # Find the subset that covers the maximum number of remaining elements
        best_subset = max(subsets, key=lambda s: len(s.intersection(universe_set)))

        # Remove the elements covered by the best subset
        universe_set -= best_subset

        # Add the best subset to the selected subsets
        selected_subsets.append(best_subset)

    return selected_subsets


def get_covering_centers(points: List[np.ndarray], max_cluster_size: float) -> List[np.ndarray]:
    centers = []
    for p in points:
        curr_centers = []
        mask = cdist(p, p) < max_cluster_size
        universe = set(range(len(p)))
        subsets = [set(mask_row.nonzero()[0].tolist()) for mask_row in mask]
        curr_cover = set_cover(universe, subsets)

        # plt.figure(figsize=(10, 10))
        # plt.xlim(0, 256)
        # plt.ylim(0, 256)
        # plt.scatter(p[:, 0], p[:, 1])
        # for i in range(len(p)):
        #     plt.annotate(f"{i}", p[i])
        for group in curr_cover:
            p_group = p[list(group)]
            # x, y, width, height = get_bounding_box(p_group)
            # rectangle = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            # plt.gca().add_patch(rectangle)
            mean_point = p_group.mean(axis=0)
            curr_centers.append(mean_point)
            # plt.scatter(*mean_point, c='green')
        centers.append(np.array(curr_centers).round().astype(int))
        # plt.show()
    return centers


def get_bounding_box(points: np.ndarray) -> Tuple[int, int, int, int]:
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return min_x, min_y, max_x - min_x, max_y - min_y

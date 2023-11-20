import numpy as np

from prime_slam.observation import ObservationData
from prime_slam.slam.mapping.map import Map
from prime_slam.slam.tracking.matching.frame_matcher import ObservationsMatcher
from prime_slam.slam.tracking.matching.map_matcher import MapMatcher

__all__ = ["SuperPointMatcher"]


class SuperPointMatcher(ObservationsMatcher, MapMatcher):
    def __init__(self, nn_thresh=0.5):
        self.nn_thresh = nn_thresh

    def match_observations(
        self, prev_observations: ObservationData, new_observations: ObservationData
    ):
        prev_descriptors = prev_observations.descriptors
        new_descriptors = new_observations.descriptors
        return self.match_superpoint(new_descriptors, prev_descriptors)

    def match_map(self, landmark_map: Map, new_observations: ObservationData):
        return self.match_superpoint(
            new_observations.descriptors,
            landmark_map.descriptors,
        )

    def match_superpoint(self, first_descriptors, second_descriptors):
        first_descriptors = first_descriptors.T
        second_descriptors = second_descriptors.T

        if first_descriptors.shape[1] == 0 or second_descriptors.shape[1] == 0:
            return np.zeros((3, 0))
        if self.nn_thresh < 0.0:
            raise ValueError("'nn_thresh' should be non-negative")
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(first_descriptors.T, second_descriptors)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < self.nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(first_descriptors.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((int(keep.sum()), 2))
        matches[:, 0] = m_idx1
        matches[:, 1] = m_idx2

        return matches.astype(int)

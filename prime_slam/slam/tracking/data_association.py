__all__ = ["DataAssociation"]


class DataAssociation:
    def __init__(self):
        self._reference_indices = {}
        self._target_indices = {}
        self._unmatched_reference_indices = {}
        self._unmatched_target_indices = {}

    def set_associations(
        self,
        observation_name,
        reference_indices,
        target_indices,
        unmatched_reference_indices,
        unmatched_target_indices,
    ):
        self._reference_indices[observation_name] = reference_indices
        self._target_indices[observation_name] = target_indices
        self._unmatched_reference_indices[
            observation_name
        ] = unmatched_reference_indices
        self._unmatched_target_indices[observation_name] = unmatched_target_indices

    def get_matched_reference(self, observation_name):
        return self._reference_indices[observation_name]

    def get_matched_target(self, observation_name):
        return self._target_indices[observation_name]

    def get_unmatched_reference(self, observation_name):
        return self._unmatched_reference_indices[observation_name]

    def get_unmatched_target(self, observation_name):
        return self._unmatched_target_indices[observation_name]

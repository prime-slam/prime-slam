from src.description.descriptor_base import Descriptor
from src.detection.detector_base import Detector
from src.observation.observations_batch import ObservationsBatch
from src.sensor.sensor_data_base import SensorData


class ObservationsCreator:
    def __init__(self, detector: Detector, descriptor: Descriptor):
        self.detector = detector
        self.descriptor = descriptor

    def create_observations(self, sensor_data: SensorData):
        keyobjects = self.detector.detect(sensor_data)
        descriptors = self.descriptor.descript(keyobjects, sensor_data)
        observations = ObservationsBatch(keyobjects, descriptors)

        return observations

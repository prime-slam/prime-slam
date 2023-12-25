import cv2
import numpy as np
import torch

from external.superglue.models.superglue import SuperGlue as SuperGlueModel
from prime_slam.observation import ObservationData
from prime_slam.slam.tracking.matching.frame_matcher import ObservationsMatcher

__all__ = ["SuperGlue"]


class SuperGlue(ObservationsMatcher):
    def __init__(self, device="cuda"):
        if device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.super_glue_matcher = (
            SuperGlueModel(SuperGlueModel.default_config).eval().to(self.device)
        )

    def match_observations(
        self, prev_observations: ObservationData, new_observations: ObservationData
    ):
        input_data = self.__preprocess_input_data(new_observations, frame_index=0)
        input_data |= self.__preprocess_input_data(prev_observations, frame_index=1)

        with torch.no_grad():
            predictions = self.super_glue_matcher(input_data)

        predictions = {k: v[0].cpu().numpy() for k, v in predictions.items()}

        matches = predictions["matches0"]
        matches = list(filter(lambda match: match[1] > -1, enumerate(matches)))
        return np.array(matches)

    def __transform_image(self, image_rgb):
        gray = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2GRAY)
        scaled_torch = (
            torch.from_numpy(gray / 255.0).float()[None, None].to(self.device)
        )
        return scaled_torch

    def __preprocess_input_data(self, observations: ObservationData, frame_index: int):
        return {
            f"keypoints{frame_index}": torch.stack(
                [
                    torch.from_numpy(observations.coordinates.astype(np.float32)).to(
                        self.device
                    )
                ]
            ),
            f"scores{frame_index}": torch.stack(
                [
                    torch.from_numpy(observations.uncertainties.astype(np.float32)).to(
                        self.device
                    )
                ]
            ),
            f"descriptors{frame_index}": torch.stack(
                [
                    torch.from_numpy(observations.descriptors.T.astype(np.float32)).to(
                        self.device
                    )
                ]
            ),
            f"image{frame_index}": self.__transform_image(
                observations.sensor_measurement.rgb.image
            ).to(self.device),
        }

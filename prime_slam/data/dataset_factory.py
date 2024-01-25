# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from prime_slam.data.data_format import DataFormatRGBD, DataFormatStereo
from prime_slam.data.hilti_stereo_dataset import HiltiStereoDataset
from prime_slam.data.icl_nuim_dataset import ICLNUIMRGBDDataset
from prime_slam.data.rgbd_dataset import RGBDDataset
from prime_slam.data.tum_rgbd_dataset import TUMRGBDDataset
from prime_slam.typing.hints import DetectionMatchingConfig

__all__ = ["DatasetFactory"]


class DatasetFactory:
    @staticmethod
    def create_from_rgbd(data_format: str, data_path: Path) -> RGBDDataset:
        formats = {
            DataFormatRGBD.tum: TUMRGBDDataset,
            DataFormatRGBD.icl: ICLNUIMRGBDDataset,
            DataFormatRGBD.icl_tum: ICLNUIMRGBDDataset.create_tum_format,
        }

        try:
            return formats[DataFormatRGBD[data_format]](data_path)
        except KeyError:
            raise ValueError(
                f"Unsupported data format {data_format}. "
                f"Expected: {DataFormatRGBD.to_string()}"
            )

    @staticmethod
    def create_from_stereo(
        data_format: str,
        data_path: Path,
        detection_matching_config: DetectionMatchingConfig,
    ) -> RGBDDataset:
        formats = {
            DataFormatStereo.hilti: HiltiStereoDataset,
        }

        try:
            return formats[DataFormatStereo[data_format]](
                data_path, detection_matching_config
            )
        except KeyError:
            raise ValueError(
                f"Unsupported data format {data_format}. "
                f"Expected: {DataFormatStereo.to_string()}"
            )

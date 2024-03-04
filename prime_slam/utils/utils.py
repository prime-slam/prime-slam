# Copyright (c) 2024, Moskalenko Ivan, Anastasiia Kornilova
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

import mrob

from pathlib import Path


def write_trajectory(path_to_save: Path, trajectory):
    with open(path_to_save, "w") as f:
        for pose in trajectory:
            qx, qy, qz, qw = mrob.geometry.so3_to_quat(pose[:3, :3])
            x, y, z = pose[:3, 3]
            f.write(f"{x} {y} {z} {qx} {qy} {qz} {qw}\n")

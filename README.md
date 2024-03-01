# prime-slam
SLAM system using various types of landmarks. README will be supplemented.

## Building Docker Image
First you need to clone the repository with all the submodules:
```bash
git clone --recursive https://github.com/prime-slam/prime-slam.git
```
Then you can build the Docker image:
```bash
docker build -t prime-slam .
```

## Running Docker Container
To run the container use the following command:
```bash
docker run --gpus all --rm -v <DATA_PATH>:/data prime-slam --configuration-path <PATH_TO_CONFIG>
```
You can see an example of a configuration file at `configs/hilti2021.yaml`.
All output data of the algorithm will be saved at `/data/output/` folder.

## Data formats
### `icl`
```
/data
├── scene_0.depth — depth in ICL format
├── scene_0.png — image
├── scene_0.txt — camera parameters
├── scene_1.depth
├── scene_1.png
├── scene_1.txt
...
├── scene_N.depth
├── scene_N.png
└── scene_N.txt
```
File names may differ from those shown above. Additional information can be found [here](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html).
### `tum`
```
/data
├── /rgb — depth in ICL format
    ├── image_0.png
    ...
    └── image_N.png
├── /depth
    ├── depth_0.png
    ...
    └── depth_M.png
└── groundtruth.txt — gt poses in TUM format
```
Additional information can be found [here](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
### `icl_tum`
ICL data presented in `tum` format.
### `stereo`
```
/data
├── /cam0 — first camera in the stereo-pair
    ├── 1630486019.540119000.png
    ...
    └── 1630486369.558611000.png
└── /cam1 — second camera in the stereo-pair
    ├── 1630486019.540119000.png
    ...
    └── 1630486369.558611000.png
```
The number of files in the folders `cam0` and `cam1` should be the same.
### `stereo_lidar`
```
/data
├── /cam0 — first camera in the stereo-pair
    ├── 1630486019.540119000.png
    ...
    └── 1630486369.558611000.png
├── /cam1 — second camera in the stereo-pair
    ├── 1630486019.540119000.png
    ...
    └── 1630486369.558611000.png
└── /pcds — point clouds
    ├── 1630486019.382846720.pcd
    ...
    └── 1630486369.606357760.pcd
```
The number of files in the folders `cam0` and `cam1` should be the same.
All file names should start with the corresponding timestamp, as this allows point clouds and color images to be synchronised.

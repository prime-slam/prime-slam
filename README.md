# prime-slam
SLAM system using various types of landmarks. README will be supplemented.

## Building Docker Image
```bash
docker build -t prime-slam .
```

## Running Docker Container
To run the container use the following command:
```bash
docker run --rm -v <DATA_PATH>:/data prime-slam [OPTIONAL_ARGS]
```
The following `[OPTIONAL_ARGS]` can be used:
```bash
optional arguments:
  -h, --help            show this help message and exit
  --data PATH, -d PATH  path to data (default: data/)
  --data-format STR, -D STR
                        data format: tum, icl, icl_tum (default: icl_tum)
  --save-cloud BOOL, -s BOOL
                        save resulting cloud (default: True)
  --cloud-save-path PATH, -S PATH
                        path to the saved cloud (default: resulting_cloud.pcd)
  --verbose BOOL, -v BOOL
                        print metrics (default: True)
```
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

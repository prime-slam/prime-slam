# prime-slam
SLAM system using various types of landmarks. README will be supplemented.

## Building Docker Image
```bash
docker build -t prime-slam .
```

## Running Docker Container
To run the container use the following command:
```bash
docker run --rm \
-v <IMAGES_PATH>:/prime_slam/rgb \
-v <DEPTH_MAPS_PATH>:/prime_slam/depth \
-v <INTRINSICS_PATH>:/prime_slam/intrinsics.txt \
-v <POSES_PATH>:/prime_slam/poses.txt \
prime-slam [OPTIONAL_ARGS]
```
The following `[OPTIONAL_ARGS]` can be used:
```bash
optional arguments:
  -h, --help            show this help message and exit
  --imgs PATH, -i PATH  path to images (default: rgb/)
  --depths PATH, -d PATH
                        path to depth maps (default: depth/)
  --intrinsics PATH, -I PATH
                        path to intrinsics file (default: intrinsics.txt)
  --poses PATH, -p PATH
                        path to gt poses (for evaluation) (default: poses.txt)
  --depth-scaler NUM, -D NUM
                        depth map scaler (default: 5000)
  --frames-step NUM, -f NUM
                        step between keyframes (default: 20)
  --save-cloud BOOL, -s BOOL
                        save resulting cloud (default: True)
  --cloud-save-path PATH, -S PATH
                        path to the saved cloud (default: resulting_cloud.pcd)
  --verbose BOOL, -v BOOL
                        print metrics (default: True)
```

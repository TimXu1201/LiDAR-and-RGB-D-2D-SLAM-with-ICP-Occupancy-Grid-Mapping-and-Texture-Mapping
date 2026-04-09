# ECE276A Project 2: 2D SLAM

# Dependencies

 pip install numpy scipy matplotlib opencv-python gtsam

# Directory Structure

Ensure your data is organized as follows relative to the script (`pr2.py`):
- `../data/` (Contains `Encoders*.npz`, `Hokuyo*.npz`, `Imu*.npz`, `Kinect*.npz`, and `dataRGBD/` folder)
- `../outputs/` (Auto-generated folder where all output PNGs will be saved)
- `icp_warm_up/data/` (Contains the .npz files for the 3D ICP warm-up)

# Warm-up: 3D ICP Registration
Runs the 3D point-to-point ICP with Z-axis (yaw) discrete initialization for `drill` and `liq_container`.

python3 pr2.py --part warmup --icp_warmup_dir icp_warm_up/data

# Part 1: Dead Reckoning (Odometry)

python3 pr2.py --part 1 --seq 20
python3 pr2.py --part 1 --seq 21

# Part 2: ICP Scan Matching

python3 pr2.py --part 2 --seq 20
python3 pr2.py --part 2 --seq 21

# Part 3: Occupancy Grid & Texture Mapping

python3 pr2.py --part 3 --seq 20
python3 pr2.py --part 3 --seq 21

# Part 4: GTSAM Pose Graph Optimization

python3 pr2.py --part 4 --seq 20
python3 pr2.py --part 4 --seq 21

# Advanced Usage: Parameter Tuning

If you need to fine-tune the SLAM performance, you can append these arguments to any command:

# ICP Parameters:
- `--scan_stride 2`: Downsample LiDAR rays to speed up ICP.
- `--icp_max_iter 20`: Maximum iterations for point-to-line ICP.
- `--max_corr 0.5`: Maximum distance (meters) to associate a point with a line.

# Loop Closure (GTSAM) Parameters:
- `--loop_radius 1.0`: Search radius (meters) for proximity-based loop closures.
- `--loop_min_sep 30`: Minimum node separation to be considered a valid loop.
- `--loop_mse_th 0.0015`: Maximum MSE threshold to accept a loop closure ICP match.

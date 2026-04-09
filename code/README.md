# 2D SLAM Code Guide

This folder contains the implementation for LiDAR and RGB-D 2D SLAM, including dead reckoning, ICP scan matching, occupancy-grid mapping, texture mapping, and graph-based pose refinement.

## Dependencies

```bash
pip install numpy scipy matplotlib opencv-python gtsam
```

## Expected Data Layout

Place the local data in the following structure relative to `pr2.py`:

- `../data/`
  Contains `Encoders*.npz`, `Hokuyo*.npz`, `Imu*.npz`, `Kinect*.npz`, and `dataRGBD/`
- `../outputs/`
  Output folder for generated PNG results
- `icp_warm_up/data/`
  Small point-cloud inputs for the ICP warm-up

## Example Commands

### Warm-up: 3D ICP Registration

```bash
python3 pr2.py --part warmup --icp_warmup_dir icp_warm_up/data
```

### Dead Reckoning

```bash
python3 pr2.py --part 1 --seq 20
python3 pr2.py --part 1 --seq 21
```

### ICP Scan Matching

```bash
python3 pr2.py --part 2 --seq 20
python3 pr2.py --part 2 --seq 21
```

### Occupancy Grid and Texture Mapping

```bash
python3 pr2.py --part 3 --seq 20
python3 pr2.py --part 3 --seq 21
```

### Pose Refinement

```bash
python3 pr2.py --part 4 --seq 20
python3 pr2.py --part 4 --seq 21
```

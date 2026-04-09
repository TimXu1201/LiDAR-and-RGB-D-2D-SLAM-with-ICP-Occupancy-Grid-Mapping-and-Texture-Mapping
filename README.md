# LiDAR and RGB-D 2D SLAM with ICP, Occupancy Grid Mapping, and Texture Mapping

This repository presents a **2D SLAM** pipeline built from wheel odometry, LiDAR scan matching, occupancy-grid mapping, RGB-D texture mapping, and graph-based trajectory refinement.

The workflow starts from dead reckoning, improves pose consistency with ICP, builds geometric and textured maps, and then refines the final trajectory with optimization.

## Project Highlights

- dead reckoning from encoder and IMU measurements
- 3D ICP warm-up on canonical object point clouds
- 2D LiDAR scan matching with point-to-line ICP
- occupancy-grid mapping from aligned range scans
- RGB-D texture mapping on the estimated floor map
- graph-based pose refinement with GTSAM

## Repository Structure

- `code/pr2.py`
  Main SLAM pipeline.
- `code/icp_warm_up/`
  ICP warm-up utilities and compact reference assets.
- `outputs/`
  Selected figures for trajectory estimation, occupancy maps, texture maps, and ICP warm-up.
- `docs/`
  Supporting robot-configuration reference material.
- `report.pdf`
  Project Description.

## Environment

Typical dependencies:

- Python 3.10+
- numpy
- scipy
- matplotlib
- opencv-python
- gtsam

## Notes

- Raw sensor logs and RGB-D datasets are not included in the public repository.
- `outputs/` is kept because the figures are useful for understanding the pipeline and final results.
- `code/pr2.py` expects a local folder layout for input data, so the dataset path structure should be recreated before rerunning.

# LiDAR and RGB-D 2D SLAM with ICP, Occupancy Grid Mapping, and Texture Mapping

This repository contains my work on **2D SLAM** using wheel odometry, LiDAR scan matching, occupancy-grid mapping, RGB-D texture mapping, and pose-graph optimization.

The project starts from dead reckoning, improves alignment with ICP, builds geometric and textured maps, and finally refines the trajectory with graph-based optimization.

## Project Highlights

- dead reckoning from encoder and IMU measurements
- 3D ICP warm-up on canonical object point clouds
- 2D LiDAR scan matching with point-to-line ICP
- occupancy-grid mapping from aligned range scans
- RGB-D texture mapping on the estimated floor map
- pose-graph optimization with GTSAM

## Repository Structure

- `code/pr2.py`
  Main SLAM pipeline.
- `code/icp_warm_up/`
  ICP warm-up utilities and small reference assets.
- `outputs/`
  Selected result figures for trajectory estimation, occupancy maps, texture maps, and ICP warm-up.
- `docs/`
  Supporting robot-configuration material.
- `276A_project2.pdf`
  Project report.

## Environment

Typical dependencies:

- Python 3.10+
- numpy
- scipy
- matplotlib
- opencv-python
- gtsam

## Notes

- The large course dataset under `data/` is excluded from the public repository.
- `outputs/` is kept because the figures are useful for portfolio presentation.
- The original development setup used local relative paths expected by `code/pr2.py`; add the dataset in the expected folder layout before rerunning.

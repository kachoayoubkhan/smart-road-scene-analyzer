# Project Overview

## Problem Statement

Road-scene images and videos contain many safety signals at the same time: vehicles, pedestrians, bicycles, lane markings, road surface quality, and the visible drivable region. Manually reviewing this information is slow and inconsistent, especially when many frames or videos need to be inspected.

## Project Goal

Smart Road Scene Analyzer is a computer vision web application that combines deep learning and classical image processing to analyze road scenes. The system detects road users with YOLO, extracts lane markings with OpenCV, approximates road regions, optionally detects potholes with a custom model, and converts these signals into a simple risk score.

## Real-World Use Cases

- Driver-assistance research prototypes
- Road safety analytics demos
- Traffic-scene inspection for academic projects
- Road damage monitoring with a custom pothole model
- Portfolio demonstration of practical deep learning and OpenCV skills

## System Architecture

```text
Input image/video
      |
      v
Streamlit dashboard
      |
      +-- YOLO object detection
      +-- OpenCV lane detection
      +-- Classical road segmentation
      +-- Optional custom pothole YOLO model
      |
      v
Risk assessment rules
      |
      v
Annotated output + markdown report
```

## Input and Output Workflow

1. The user uploads a road image or video.
2. The dashboard reads the file and converts image color channels consistently.
3. The selected modules run on the image or sampled video frames.
4. The processed output is displayed with overlays and bounding boxes.
5. The risk module calculates a score from 0 to 100.
6. The user can download the processed media and a markdown report.

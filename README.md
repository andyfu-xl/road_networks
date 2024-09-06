# Road Networks

This repository contains the code and resources for an MSc honours project: **Combining Graph Reinforcement Learning and Trajectory Prediction for Vehicular Ad Hoc Network Routing**. The project investigates a proactive routing system, which predicts the VANET's dynamic with trajectory prediction. 

The repository consists of two test environments, both are retrieved from [OpenStreetMap](https://www.openstreetmap.org/). Two road networks are parts of the Edinburgh city, the smaller map (__Edinburgh__) was only used for trajectory prediction and the larger map (__Larger_Edinburgh__) was used for all experiments.

## Table of Contents

- [Getting started](#Getting-started)
- [For Markers](#For-Markers)

## Getting Started
The project was developed and tested on Ubuntu 20.04.6 LTS (64-bit) only.

### The Simulator
[Simulation of Urban MObility](https://eclipse.dev/sumo/) (__SUMO__) was involved in this project, please install the simulator before running, click [here](https://sumo.dlr.de/docs/Downloads.php) for the official installation guideline.

### Requirements
The project was developed with Python 3.10.12, and managed via [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

First, clone the repository to your local machine:
```bash
git clone https://github.com/andyfu-xl/road_networks.git
cd road_networks
```

Create a conda environment using the provided `requirements.txt`:
```bash
conda create --name your-env-name --file requirements.txt
conda activate your-env-name
```


## For Markers:
This project was developed and tested exclusively on Ubuntu due to its advantages for machine learning tasks. The code may not run directly on other operating systems without modification. However, all experimental results are included alongside the implementation to facilitate inspection and review
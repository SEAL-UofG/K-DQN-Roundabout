# KAN-DRL-Roundabout

```
# K-DQN Roundabout Navigation

This repository contains the implementation of the KAN-based, conflict-avoidance, and proper-lane-detection DRL system for autonomous vehicle navigation in roundabouts, as described in the paper "Complex Interactive Driving in a Roundabout: A Conflicts-free and Speed-lossless Decision System Based on KAN-assisted Deep Reinforcement Learning".

## Overview

The proposed system employs a deep Q-learning network (DQN) enhanced with a Kolmogorov-Arnold network (KAN) to learn safe and efficient driving strategies in complex multi-vehicle roundabouts. The system also includes an action inspector to avoid collisions and a route planner to enhance driving efficiency and safety.

## Features

- KAN-enhanced DQN for robust and precise learning of driving strategies
- Action inspector to replace dangerous actions and avoid collisions
- Route planner to optimize lane selection and driving efficiency
- Model predictive control (MPC) for stable and precise execution of driving actions

## Requirements

- Python 3.6
- PyTorch 1.10
- This work based on Highway-env. You can find details at https://github.com/Farama-Foundation/HighwayEnv.


## Installation

1. Clone the repository:
```

git clone https://github.com/SEAL-UofG/K-DQN-Roundabout.git

```
Copy
2. Install the required dependencies:
```

pip install -r requirements.txt

```
## Usage

1. Configure the desired training and validation settings in the configuration file.

2. Run the training script:
```

python run_kan.py



```

## Results

The proposed system demonstrates superior performance in terms of safety and efficiency compared to traditional benchmark algorithms. It achieves lower collision rates, reduced travel times, and faster training convergence across various traffic flow scenarios.

```



```
## License

This project is licensed under the [MIT License](LICENSE).
```
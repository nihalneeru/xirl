# X-MAGICAL Dataset

This is the [X-MAGICAL](https://github.com/kevinzakka/x-magical) dataset, created for the publication [XIRL: Cross-embodiment Inverse Reinforcement Learning](https://x-irl.github.io/).

If you use this dataset, please cite [our paper](https://x-irl.github.io/):

```bibtex
@article{zakka2021xirl,
    title = {XIRL: Cross-embodiment Inverse Reinforcement Learning},
    author = {Zakka, Kevin and Zeng, Andy and Florence, Pete and Tompson, Jonathan and Bohg, Jeannette and Dwibedi, Debidatta},
    journal = {Conference on Robot Learning (CoRL)},
    year = {2021}
}
```

## Data Collection

We train a reinforcement learning policy for each agent embodiment with SAC for `1M` steps. The reward is the fraction of debris swept inside the goal zone. We use the default hyper parameters from [Yarats et al.](https://github.com/denisyarats/pytorch_sac) with a frame stack value of 3 and an action repeat value of 1. Once training is complete, we rollout the policy to generate a dataset of 1000 demonstrations, discarding any rollouts that are unsuccessful.

## Directory Structure

```bash
├── gripper
│   ├── 0
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── ...
│   │   ├── actions.json
│   │   ├── rewards.json
│   │   └── states.json
│   └── 1
│   └── ...
│   └── 999
└── longstick
└── mediumstick
└── shortstick
```

## Setup

You'll need `numpy` and `Pillow` to load and manipulate the demonstrations in the dataset. If you don't already have them, you can install them via:

```bash
pip install -r requirements.txt
```

## Demonstration Example

Here is some Python code for loading and manipulating the observation, action, reward and state tuples in a demonstration:

```python
import json
import numpy as np
from PIL import Image

# Suppose we are in the 'gripper/0/' directory.
img = np.asarray(Image.open('0.png'))
img.shape  # (384, 384, 3).
img.dtype  # dtype('uint8').

# Suppose there are 15 PNG images, i.e. the total demonstration
# length is 15 timesteps.

actions = np.asarray(json.load(open('actions.json', 'r')))
actions.shape  # (15, 3) if 'gripper' else (15, 2).
actions.dtype  # dtype('float64').

rewards = np.asarray(json.load(open('rewards.json', 'r')))
rewards.shape  # (15,).
rewards.dtype  # dtype('float64').

states = np.asarray(json.load(open('states.json', 'r')))
states.shape  # (15, 51) if 'gripper' else (15, 48).
states.dtype  # dtype('float64').
```

## License

This data is licensed by Google Inc. under a Creative Commons Attribution 4.0 International License.

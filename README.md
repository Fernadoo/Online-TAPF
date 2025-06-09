# Online TAPF


### Set up

1. Use `conda` to install python,

```bash
$ conda create -n marp python=3.9.18
```

2. Activate the conda environment and install the dependencies,

```bash
$ conda activate marp
$ pip install -r requirements.txt
```



### Usage

Some live demos can be found in the `figs/` folder.

To run the code, please first view the manual,

```bash
python run_maituan_ta.py -h
```

It will output the detailed usage of the program,

```bash
usage: run_maituan_ta.py [-h] [--router ROUTER] [--num NUM] [--assigner ASSIGNER] [--alpha ALPHA] [--model MODEL] [--task-seed TASK_SEED] [--robot-seed ROBOT_SEED] [--vis]
                         [--save SAVE]

Multi-Agent Task Assignment and Route Planning.

optional arguments:
  -h, --help            show this help message and exit
  --router ROUTER       choose an algorithm among [libiao, tpts, rhcr, ...]
  --num NUM             Specify the number of agents
  --assigner ASSIGNER   Choose an assigner among [closer, alpha, mpc, ...]
  --alpha ALPHA         Choose a threshold for the adaptive assigner
  --model MODEL         Choose a pretrained model among [0, 1, 2, 3]
  --task-seed TASK_SEED
                        Specify a seed for task radomization
  --robot-seed ROBOT_SEED
                        Specify a seed for robot radomization
  --vis                 Visulize the process
  --save SAVE           Specify the path to save the animation
```

For example,

```bash
python run_maituan_ta.py --num 50 --router libiao --assigner closer --task-seed 0 --robot-seed 0 --vis --save figs/demo.mp4
```

where

- `--num` stands for the number of robots, here 50;
- `--router` stands for the routing algorithm being used; in this case, it is a planning algorithm based on the touring rule (although during the development process it was named `libiao` by some historical reason);
- `--assigner` represents the task assignment algorithm being used; in this case, it is the "closer first" assignment algorithm;
- `--task-seed` represents the random seed for the order of product shelving, set to 0 here;
- `--robot-seed` represents the random seed for the initial positions of the robots at startup, set to 0 here;
- `--vis` indicates whether to enable the visualization interface;
- `--save` indicates the path to save the animation video, which is `figs/demo.mp4` here.

### Optional Algorithms

We provide several options for `--router` and `--assigner`.

##### Path Planning Algorithms

| `--router` Option | Corresponding Algorithm in the Paper |
| ----------------- | ------------------------------------ |
| libiao            | Touring-with-early-exit              |
| tpts              | PP-$h_{slow}$                        |
| tpts_r            | PP-$h_{fast}$                        |
| rhcr              | RHCR-$h_{slow}$                      |
| rhcr_r            | RHCR-$h_{fast}$                      |

##### Task Assignment Algorithms

| `--assigner` Option | Corresponding Algorithm in the Paper | Secondary Options | Notes                               |
| ----------------- | ------------------------------------- | ----------------- | ----------------------------------- |
| closer            | Stateless assignment                  |                   |                                     |
| farther           | Stateless assignment                  |                   |                                     |
| random            | Stateless assignment                  |                   |                                     |
| alpha             | Adaptive assignment                   | `--alpha`         | For example, `--alpha 0.235`       |
| mpc               | Predictive assignment                 | `--model`         | Listed in `./assigner/mpc.py:line13` |


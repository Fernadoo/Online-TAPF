# MARP
> A simulator for multi-agent task assignment and path finding 
>
> 仿真环境详细文档参见https://marp.readthedocs.io/en/latest/



### 安装流程

1. 用conda安装对应版本python，

```bash
$ conda create -n marp python=3.9.18
```

2. 激活虚拟环境以后安装依赖库，

```bash
$ conda activate marp
$ pip install -r requirements.txt
```



### 使用方法

运行以下命令即可查看使用方式，

```bash
python run_maituan_ta.py -h
```

将会输出命令行传参的详细说明，

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

例如，

```bash
python run_maituan_ta.py --num 50 --router libiao --assigner closer --task-seed 0 --robot-seed 0 --vis --save figs/demo.mp4
```

其中，

- `--num`代表机器人的数量，此处为50；
- `--rounter`代表采用的路径规划算法，此处为基于巡回规则的规划算法（虽然开发过程将其取名为libiao，意为以libiao为代表的基于交通规则的算法）；
- `--assigner`代表采用的任务分配算法，此处为“近端优先”分配算法；
- `--task-seed`代表货品上架顺序的随机种子，此处设为0；
- `--robot-seed`代表启动时机器人初始位置的随机种子，此处设为0；
- `--vis`代表是否开启可视化界面；
- `--save`代表存储动画视频，此处路径为`figs/demo.mp4`



### 可选算法

我们为`--router` 和 `--assigner`提供了若干待选项。

##### 路径规划算法

| `--router`选项 | 与报告中对应的算法      |
| -------------- | ----------------------- |
| libiao         | Touring-with-early-exit |
| tpts           | PP-$h_{slow}$           |
| tpts_r         | PP-$h_{fast}$           |
| rhcr           | RHCR-$h_{slow}$         |
| rhcr_r         | RHCR-$h_{fast}$         |

##### 任务分配算法

| `--assign`选项 | 与报告中对应的算法 | 二级选项  | 备注                               |
| -------------- | ------------------ | --------- | ---------------------------------- |
| closer         | 无状态分配         |           |                                    |
| farther        | 无状态分配         |           |                                    |
| random         | 无状态分配         |           |                                    |
| alpha          | 自适应分配         | `--alpha` | 例如`--alpha 0.235`                |
| mpc            | 预测性分配         | `--model` | 罗列在了`./assigner/mpc.py:line13` |


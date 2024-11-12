# Code Instruction for CTPG

This repository hosts the PyTorch implementation of "**Efficient Multi-Task Reinforcement Learning with Cross-Task Policy Guidance**" (CTPG) on two benchmarks: **HalfCheetah Locomotion Benchmark** and **MetaWorld Manipulation Benchmark**.

**NOTE**:

The code is based on the [MTRL](https://github.com/facebookresearch/mtrl) codebase.

The [HalfCheetah Locomotion Benchmark](https://breakend.github.io/gym-extensions/) is already integrated into the code and does not require additional installation.

The [MetaWorld Manipulation Benchmark](https://meta-world.github.io/) requires extra installation. Since MetaWorld is under active development, all experiments are performed on the stable release version v2.0.0: https://github.com/Farama-Foundation/Metaworld/tree/v2.0.0.



## Setup

1. Set up the working environment: 

```shell
pip install -r requirements.txt
```

2. Set up the MetaWorld benchmark: 

First, install the mujoco-py package by following the [instructions](https://github.com/openai/mujoco-py#install-mujoco).

Then, install MetaWorld:

```shell
pip install git+https://github.com/Farama-Foundation/Metaworld.git@v2.0.0
```



## Training

Use the `scripts/start.sh` script to quickly run the code as follows:

```shell
bash scripts/start.sh $alg $env $map
```

- `$alg` includes: `guide_mtsac`, `guide_mhsac`, `guide_pcgrad`, `guide_sm`, `guide_paco`
- `$env` includes: `metaworld` and `gym_extensions`
- `$map` includes:
  - `mt10`, `mt50` (for `metaworld`)
  - `halfcheetah_gravity-mt5`, `halfcheetah_body-mt8` (for `gym_extensions`)

For example, to run `MHSAC w/ CTPG` on the `MetaWorld-MT10` setup:

```shell
bash scripts/start.sh guide_mhsac metaworld mt10
```

All results will be saved in the `log` folder.



## See Also

Refer to [MTRL](https://github.com/facebookresearch/mtrl), [Gym-extensions](https://github.com/Breakend/gym-extensions), [MetaWorld](https://github.com/Farama-Foundation/Metaworld), [mujoco-py](https://github.com/openai/mujoco-py) for additional instructions.
# Ad Asset Portfolio Optimization via Policy Learning
This repository contains the code used for the experiments in ["Ad Asset Portfolio Optimization via Policy Learning"](http://papers.adkdd.org/2026/papers/adkdd26-tanaka-ad.pdf) by [Koichi Tanaka](https://tana-51.github.io), Zhi Wang, Masahiro Asami, Kota Ishizuka, Kosuke Kawakami, Yuta Saito. This paper was accepted at [AdKDD 2026](https://www.adkdd.org).

## Abstract
In online advertising, it is crucial to identify optimal ad assets for diverse users to maximize platform revenue. Many algorithms have been developed for selecting personalized ad assets, and have been applied in industrial systems. However, to fully leverage the performance of these algorithms, advertisers are required to submit an appropriate set of ad assets, which we call an ad asset portfolio, to the platforms. This problem, which we call ad asset portfolio optimization (Ad-POP), is a crucial issue for online advertising, but it remains underexplored in the existing literature.
To tackle this problem, we first formulate Ad-POP in the contextual combinatorial bandits, and reduce it to an off-policy learning problem (OPL), which aims to optimize a new policy solely using historical logged data collected by a different policy. 
Typical OPL methods can be applied to the Ad-POP problem and categorized into two types of approaches: the independent approach and the exact approach. 
The independent approach constructs an ad asset portfolio by adding ads with the highest value, while the exact approach treats an ad set as a single action and directly optimizes ad asset portfolios. However, these approaches suffer from high bias and high variance, respectively.
To address these challenges, we propose a novel algorithm, named Ad Asset Portfolio Optimization via Meta Information (Ad-POM). Our algorithm decomposes an ad selection policy into a first-stage policy for selecting meta information, such as the size of the portfolio and the degree of diversity, and a second-stage policy for selecting the portfolio given the meta information. Specifically, we propose a new policy gradient estimator to learn the first-stage policy. This method can achieve stable optimization since it applies importance weighting only to meta information. 
Our comprehensive experiments on real-world data demonstrate that the proposed method can provide substantial improvements in Ad-POP, where existing methods fail due to the large action space and the interactions among ad assets.

## Citation
```
coming soon
```

## Setup
The Python environment is built using uv. You can build the same environment as in our experiments by cloning the repository and running uv sync directly under the folder.
```
# build the environment with uv
uv sync

# activate the environment
source .venv/bin/activate
```

## Data Preparation
This repository expects the [CreativeRanking dataset](https://arxiv.org/abs/2102.04033) to be placed under the `list/` directory at the project root. In particular, the experiment code reads the logged data from:

```
list/train_data_list.txt
```

Please create the `list/` and `images/` directories and place the CreativeRanking data files in it before running the experiments. The expected directory structure is:

```
.
|-- list/
|   `-- train_data_list.txt
|-- images/
|-- src/
|-- conf/
`-- run.sh
```

Before running the experiments, please generate `image_features.pkl` by running the
preprocessing script:

```
uv run python pre_process/autoencoder.py
```

This script extracts CLIP image embeddings, compresses them with PCA,
and saves the resulting feature dictionary as `image_features.pkl` in the project
root.

## Running Experiments
The experiments are implemented as Hydra-based Python scripts in the `src/` directory. The main experimental scripts are:

| Script | Description |
| --- | --- |
| `src/main_group_effect.py` | Evaluates the effect of varying the group-effect coefficient. |
| `src/main_cost_effect.py` | Evaluates the effect of varying the cost-effect coefficient. |
| `src/main_n_creative.py` | Evaluates the effect of varying the number of available ad assets. |
| `src/main_beta.py` | Evaluates the effect of varying the logging-policy parameter. |
| `src/main_beta1.py` | Evaluates the effect of varying the first-stage logging-policy parameter. |
| `src/main_random_sample_linspace.py` | Evaluates the effect of varying sampled meta-information candidates. |

You can run all experiments using:

```
bash run.sh
```

Each experiment can also be run individually. For example:

```
uv run ./src/main_group_effect.py \
    setting.wandb.train_name="group_effect_coeff" \
    setting.base_setting.use_wandb=False \
    setting.base_setting.num_runs=25
```

The results are saved under the `result/` directory.

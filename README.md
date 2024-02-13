# Reward Design for Justifiable Sequential Decision-Making

This repository contains the code and instructions necessary to replicate the results from the ICLR 2024 paper "[Reward Design for Justifiable Sequential Decision-Making](https://openreview.net/forum?id=OUkZXbbwQr)".

## Dependencies

This code depends on Python 3.9.13. The rest of dependencies are specified in the `requirements.txt` file. To install them, it suffices to run `pip install -r requirements.txt`. Alternatively, we have also provided a Docker image in `docker/argo/Dockerfile` which you are welcome to use. In its previous life, this project was named "Argumentative Optimization", hence the top-module uses its abbreviation, `argo`.

## Sepsis Dataset

To reproduce the results reported in the paper, there are several steps to be followed, outlined in subsequent sections.

### Accessing MIMIC-III

We use version 1.4 of [MIMIC-III dataset](https://physionet.org/content/mimiciii/1.4/) stored in a local PostgreSQL instance. We have included a Docker image in `docker/mimic/Dockerfile` that includes the necessary dependencies to load the MIMIC-III database. To setup MIMIC-III in a local PostgreSQL instance, follow the steps:
1. Review the variables and volumes defined in the `docker/mimic/docker-compose.yml` file and ensure they have the correct values for your system;
2. Build the image by executing `docker compose -f docker/mimic/docker-compose.yml build`. Note that for hosting the entire MIMIC-III dataset, you will need around 100GB of disk space;
3. Execute the following command to start the data import procedure `docker exec <container-id> /tmp/mimic-code/init.sh`.

The entire procedure will most likely take several hours to complete, after which you will have a fully initialized PostgreSQL instance containing the entire MIMIC-III dataset.

### Patient Cohort Extraction

To extract and preprocess the dataset, we rely on the [Microsoft's sepsis cohort extraction](https://github.com/microsoft/mimic_sepsis) script. We refer the reader to instructions provided in the linked repository. The output of this procedure are two *.csv files containing normalized and raw patient data. These files are an expected input to our dataset generation script, outlined in the following section.

### Create Datasets

To create dataset splits, run the following:
```bash
python -m argo.scripts.generate_dataset \
    --artifacts-dir ./assets/data/sepsis/ \
    --sepsis-cohort ./assets/data/sepsis/sepsis_final_data_normalized.csv \
    --train-chunk 0.7 \
    --val-chunk 0.5 \
    --seed 5568
```
This will create train, val and test tensor dictionaries. For a detailed usage and description of available options, run `python scripts/generate_dataset.py --help`. 

### Folder Structure

Although you're free to use any folder structure for your trained models, the default `guild.yml` configuration expects everything to be stored in the `assets` directory, with the following structure:
- `assets/data/sepsis` contains all data-related files (e.g., CSV files, preference datasets, etc.);
- `assets/models/argumentator` is expected to contain subdirectories named $4$ and $6$ designating argumentative agents (and their confusers) trained with $4$ and $6$ arguments respectively. The naming scheme is as follows: for argumentative agents, exported file name is `argumentator.<suffix>.pt` where suffix is `isolated`, `debate-minimax` or `debate-selfplay`. For confuser agents, the exported file name is `confuser.<suffix>.pt`, using already mentioned suffixes;
- `assets/models/clinician` is expected to have a clinician's policy named `clinician.pt`;
- `assets/models/judge` is expected to have three judge models, namely `judge.pt` (for $6$ arguments), `judge.4.pt` (for $4$) and `judge.aligned.pt` for a full-state judge;
- `assets/models/protagonist/ddqn` is expected to have two subdirectories, namely `full_state` and `justifiable` representing policies trained with state-based and debate-based feedback respectively. Within each subdirectory, a separate directory named `l<suffix>` where suffix is `00`, `25`, `50`, `75` or `100` contains training results of policies with different values of parameter $\lambda$.

## Training

With initialized data, we turn our attention to training individual models which we describe in the following subsections.

### Experiment Tracking

We rely on [Guild AI](https://guild.ai) tool to automate and track our experiments. The reader is encouraged to familiarize itself with the provided `guild.yml` file which defines each experiment in this paper. Because some experiments are interconnected, we describe here the exact order in which they need to be run. Before invoking the training scripts, consult the `guild.yml` file to ensure all of the necessary file dependencies are met.

### 1. Clinician Policy

To train the clinician policy used during weighted importance sampling (WIS) evaluation, it suffices to run:

```bash
guild run clinician:train
```

### 2. Judge 

To train the judge model, it suffices to run:

```bash
guild run judge:train
```

This script will additionally create the preference dataset $\mathcal{D}$ from the paper (see generated artifacts `train_preferences.pt`, `val_preferences.pt` and `test_preferences.pt`). These preferences are then used in all reported experiments.

### 3. Isolated Argumentative Agent

To train the *isolated* argumentator, it suffices to run:

```bash
guild run argumentator:train
```

We refer the reader to `guild.yml` file where further details.

### 4. Debate Agents

To train the *self-play* argumentator, it suffices to run:

```bash
guild run debate:train
```

Likewise, to train the *maxmin* version, simply replace `train` suffix with `train-minimax`. We refer the reader to `guild.yml` file for further details.

### 5. Confuser Agents

To train the *confuser* agent, first update the `guild.yml` file to specify its opponent (i.e., one of the agents trained in the previous two sections). Then, it suffices to run:

```bash
guild run confuser:train
```

### 6. Baseline and Justifiable Agents

#### Baseline Agent

To train a baseline agent, first refer to the `guild.yml` configuration and ensure all dependencies are met. In addition, set the `lmbd-justifiability` parameter to $0$ ($\lambda$ from the paper). Then, it suffices to run:

```bash
guild run protagonist-ddqn:train
```

The command will train the baseline agent using $5$ random seeds.

#### Justifiable Agent

To train the justifiable agent, first change the `lmbd-justifiability` to a desired value and ensure you pass the path to the baseline policy trained in the previous section using an argument `baseline-path`. Also, ensure any lambda-specific hyperparameters (in particular, just one, namely `n-estimation-step`) is properly set for that particular lambda value (see App. C.3 of the paper).

## Evaluation

After training all agents, we perform their evaluation in two notebooks. First, to evaluate argumentative agents, run the `notebooks/eval_argumentation.ipynb` notebook. Next, to evaluate sepsis agents, run the `notebooks/eval_protagonist.ipynb` notebook. These two notebooks will generate a bunch of *.csv files that will be stored in the `results/` directory. To generate plots, it suffices to run `notebooks/plots.ipynb` notebook.

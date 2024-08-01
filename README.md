# Cell Counting Collaboration
Between Wellcome Sanger Institute in Cambridge, the Computational Cell Analytics Group at the University of Göttingen and the Biomedical Computer Vision Lab at the University of Tartu.

Localizing Cells in Phase-Contrast Microscopy Images using Sparse and Noisy Center-Point Annotations

Based on [my (Benjamin Eckhardt's) Bachelor's Thesis in Computer Science](https://github.com/beijn/bachelor-thesis).

If you use this work please credit the original author (Benjamin Eckhardt, 2024) and cite [the Thesis](https://github.com/beijn/bachelor-thesis).

# CellNet Documentation

Note: All instructions are relative to the repo root. 

## Using the Release

### Installation
`pip install git+https://github.com/beijn/cellnet`

### Usage via Command Line
```bash
python cellnet/release.py [list of image (/folder) paths]
```
Will store the counts in `counts.json` and images with the predictions in `plots/`.
### Usage as a Python Module
```python
from cellnet import init_model, count
model = init_model(<version>)
counts, plots = count(<list_of_image_file_descriptors>, model)
```
- `counts` and `plots` are image-path indexed dictionaries.
- `<version>`: the model version
  - `'latest'`: will download and cache the latest compatible model to `~/.cache/cellnet`
  - `None`: will use what ever is already in the cache or default to `'latest'`
  - any other string will download that version from the [GitHub releases](https://github.com/beijn/cellnet/releases) 


### Trouble Shooting
#### Can't automatically determine latest model version due to GitHub API rate limiting
- provide a manual version string to `init_model(<version>)`
- or provide a `GITHUB_TOKEN` environment variable with a GH PAT with scope Content: read (eg. in [bin/init_shell](./bin/init_shell))

## Training and Releasing
Workflow Overview: 
1. [Setup](#setup)
2. [Data Setup](#data-setup)
2. [Training](#training)
3. [Releasing](#releasing)

### Setup
```bash
git clone git@github.com:beijn/cellnet.git
cd cellnet
git switch draft  # optional: development branch
mircomamba create -yf cellnet.yml || conda env create -yf cellnet.yml
micromanba activate cellnet || conda activate cellnet
bash bin/install  # editable pip install of the cellnet package
```

### Data Setup
- Create and populate the [`data`](./data) folder.
- Interactively run [`pre_images.py`](./pre_images.py) and [`pre_points.py`](./pre_points.py) to prepare the data for the model.

[`pre_images.py`](./pre_images.py) for the images in [`data/images`](./data/images) creates binary foreground/background masks in [`data/cache/fgmasks`](./data/cache/fgmasks) which help during training. Currently uses an unlicensed third-party model, which will not adapt to new image modalities and **should be replaced**. Their purpose is to include background in the training data, whereas foreground regions without point annotations are excluded. Please inspect the generated masks manually and update the `fgmask_status` column in [`data/data_quality.csv`](./data/data_quality.csv) accordingly with `ok` or `BAD`.

[`pre_points.py`](./pre_points.py) converts the label-studio `<some name>..<ANNOTATOR>.json` point annotations in [`data`](./data) to `[x,y,label-id]` arrays in [`data/cache/points`](./data/cache/points). The annotators contributing to an image are saved to [`data/cache/annotators.json`](./data/cache/annotators.json). Merging annotations of different annotators is currently not implemeted. Draft code for filtering only 'agreeing' annotations is in [`utility/agreement-ac-bo.py`](./utility/agreement-ac-bo.py). Please manually update the `annotation_status` column in [`data/data_quality.csv`](./data/data_quality.csv) accordingly with `empty`, `sparse`, `fully`, `NISSING` or `CONFLICT`.

Images with `BAD` masks, which are neither `empty` nor `fully` annotated are excluded from training. As well as images with `NISSING` or `CONFLICT`ing annotations.

### Training
```bash
bin/submit-job [remote] sbatch|local <notebook> <experiment> <release-mode>
```
[bin/submit-job](./bin/submit-job) creates a snapshot copy of the current repository state and submits a job to run the `<notebook>` with the specified `<experiment>` and `<release-mode>` settings. The resulting notebook and all its outputs are saved under [`results/<notebook>/STARTDATE-<experiment>-<release-mode>`](./results/train).
- if `remote`, the notebook will be run on the remote cluster
- `sbatch` will submit a job to run the notebook to slurm's sbatch
- `local` will run the notebook in the current shell
- `<notebook>` should be without the `.py` or `.ipynb` extension
- `<experiment>` defines experiment settings in train.py
- `<release-mode>`: `draft`: for interactive sessions; `release`: create release artifacts; `crossval`: perform cross-validation; everything else defaults to crossval and can therefore be used as free-form tag

The actual job is defined in [bin/job-ipynb](./bin/job-ipynb). It activates a matching conda environment with the help of [bin/init_conda](./bin/init_conda), executes the notebook with the specified settings using [jupytext](https://jupytext.readthedocs.io/en/latest/), and afterwards cleans up the snapshot copy for all files that have not been modified or created during the job (aka results and release artifacts).

Slurm-cluster specific settings are defined in the `#SBATCH` header of [bin/job-ipynb](./bin/job-ipynb) and (are overwritten with) the `SBATCH_ARGS` defined via [`source bin/init_shell`](./bin/init_shell). 

To modify the [`train.py`](./train.py) please consider parameterizing the new functionality via a setting in `train.CFG` and adding a corresponding experiment setting in `train.EXPERIMENTS`.

#### Examples
- `bin/submit-job remote sbatch train sigma crossval` will run the `train` notebook with the `sigma` experiment settings in cross-validation mode with slurm's sbatch on the remote cluster.
- `bin/submit-job local train default release` will run the `train` notebook with the `default` settings in the current shell and save the model release artifacts.

### Releasing
```bash
bin/release <path_to_model> [<version-tag>]
```
Will package the release artifacts for the model in the specified path and push it to the [GitHub releases](https://github.com/beijn/cellnet/releases) with the name `<model_api_version>-<version-tag>`.
- The model API version is drawn from `cellnet.__init__.__model_api_version__`.
- Pushing a release requires [`gh`, the GitHub CLI](https://cli.github.com/) to be installed and authenticated.

#### Examples
- `bin/release results/train/250229-246060-default-release` will create a release named `2-250229`
- `bin/release results/train/250229-246061-xnorm_per_channel-release dynnorm` will create a release named `2-250229-dynnorm`



# Future Directions 

Also refer to [the Thesis](https://github.com/beijn/bachelor-thesis/blob/main/Bachelor's%20Thesis.pdf)

## Machine Learning Research
- **replace the method to generate the foreground masks in `pre_images.py` with something that can be easily adapted to new datasets.**
- merge agreeing/disagreeing annotations (see draft code in [`utility/agreement-ac-bo.py`](./utility/agreement-ac-bo.py))
  - only include aggrement annotations
  - subtract neighborhoods of disagreeing annotations from the 24px radius positive point mask
- compression-less image file format
- explore more sophisticated model architectures 
- implement a better loss function, count-based loss, or best: MESA from [Lempitsky et al., 2010](https://proceedings.neurips.cc/paper_files/paper/2010/file/fe73f687e5bc5280214e0486b273a5f9-Paper.pdf).
  - or see https://arxiv.org/pdf/1907.02336
- better heatmap with repulsive distributions to better distinguish individual cells (has to keep norm property)
  - inspire by force field of two electrons in physics? (retains norm property)
  - inverse distance: https://davidborland.github.io/webpage/pdfs/MICCAI%20MOVI%202022.pdf (damages nice gaussian norm property)
- label smoothing?
- blob detection for discretized localization


## Software Engineering 
- replace [segmentation_models_pytorch](https://smp.readthedocs.io/en/latest/models.html) with a better maintained and further developed model library
- switch from segmentation_models_pytorch's save/load to [onnx](https://onnx.ai/) for superiour portability and flexibility in decoupling training and deployment, less release dependencies
- consider switching from my custom `bin/submit-job` and `bin/job-ipynb` with `experiments` configuration in the notebook to [Hydra](https://hydra.cc/docs/intro/) or so for greater standardization
- consider switching from my custom `train.py` training loop to [MosaicML' Composer](https://docs.mosaicml.com/projects/composer/en/stable/index.html) or [Lightning](https://lightning.ai/docs/pytorch/stable/) or so for more low-hanging fruits it optimizing the training process

<div align="center">

# AlphaFold 3 - PyTorch Lightning + Hydra

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/s41586-024-07487-w) -->

<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

<img src="./img/alphafold3.png" width="600">

</div>

## Description

Implementation of <a href="https://www.nature.com/articles/s41586-024-07487-w">AlphaFold 3</a> in PyTorch Lightning + Hydra

You can chat with other researchers about this work <a href="https://discord.gg/x6FuzQPQXY">here</a>

<a href="https://www.youtube.com/watch?v=qjFgthkKxcA">Review of the paper</a> by <a href="https://x.com/sokrypton">Sergey</a>

A fork with full Lightning + Hydra support is being maintained by <a href="https://github.com/amorehead">Alex</a> at <a href="https://github.com/amorehead/alphafold3-pytorch-lightning-hydra">this repository</a>

## Appreciation

- <a href="https://github.com/joseph-c-kim">Joseph</a> for contributing the Relative Positional Encoding and the Smooth LDDT Loss!

- <a href="https://github.com/engelberger">Felipe</a> for contributing Weighted Rigid Align, Express Coordinates In Frame, Compute Alignment Error, and Centre Random Augmentation modules!

- <a href="https://github.com/amorehead">Alex</a> for fixing various issues in the transcribed algorithms

- <a href="https://github.com/gitabtion">Heng</a> for pointing out inconsistencies with the paper and pull requesting the solutions

- <a href="https://github.com/gitabtion">Heng</a> for finding an issue with the molecular atom indices for the distogram loss

- <a href="https://github.com/luwei0917">Wei Lu</a> for catching a few erroneous hyperparameters

- <a href="https://github.com/amorehead">Alex</a> for the PDB dataset preparation script!

- <a href="https://github.com/patrick-kidger">Patrick</a> for <a href="https://docs.kidger.site/jaxtyping/">jaxtyping</a>, <a href="https://github.com/fferflo">Florian</a> for <a href="https://github.com/fferflo/einx">einx</a>, and of course, <a href="https://github.com/arogozhnikov">Alex</a> for <a href="https://einops.rocks/">einops</a>

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [For developers](#for-developers)
- [Citations](#citations)

## Installation

<details>

### Pip

```bash
pip install alphafold3-pytorch-lightning-hydra
```

### Conda

Install `mamba` for dependency management (as a fast alternative to Anaconda):

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Install dependencies:

```bash
# Clone project
git clone https://github.com/amorehead/alphafold3-pytorch-lightning-hydra
cd alphafold3-pytorch-lightning-hydra

# Create Conda environment
mamba env create -f environment.yaml
conda activate alphafold3-plh  # note: one still needs to use `conda` to (de)activate environments

# Install local project as package
pip3 install -e .
```

### Docker
The included `Dockerfile` contains the required dependencies to run the package and to train/inference using PyTorch with GPUs.

The default base image is `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` and installs the latest version of this package from the `main` GitHub branch.

```bash
# Clone project
git clone https://github.com/amorehead/alphafold3-pytorch-lightning-hydra
cd alphafold3-pytorch-lightning-hydra

# Build Docker container
docker build -t af3 .
```

Alternatively, use build arguments to rebuild the image with different software versions:
- `PYTORCH_TAG`: Changes the base image and thus builds with different PyTorch, CUDA, and/or cuDNN versions.
- `GIT_TAG`: Changes the tag of this repo to clone and install the package.

For example:
```bash
## Use build argument to change versions
docker build --build-arg "PYTORCH_TAG=2.2.1-cuda12.1-cudnn8-devel" --build-arg "GIT_TAG=0.1.15" -t af3 .
```

Then, run the container with GPUs and mount a local volume (for training) using the following command:

```bash
## Run Container
docker run -v .:/data --gpus all -it af3
```

</details>

## Usage

<details>

```python
import torch
from alphafold3_pytorch import Alphafold3

alphafold3 = Alphafold3(
    dim_atom_inputs = 77,
    dim_template_feats = 44
)

# Mock inputs

seq_len = 16
molecule_atom_lens = torch.randint(1, 3, (2, seq_len))
atom_seq_len = molecule_atom_lens.sum(dim = -1).amax()

atom_inputs = torch.randn(2, atom_seq_len, 77)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)

additional_molecule_feats = torch.randn(2, seq_len, 5)
is_molecule_types = torch.randint(0, 2, (2, seq_len)).bool()
molecule_ids = torch.randint(0, 32, (2, seq_len))

template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 64)
msa_mask = torch.ones((2, 7)).bool()

# Required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3)
molecule_atom_indices = molecule_atom_lens - 1  # last atom, as an example

distance_labels = torch.randint(0, 37, (2, seq_len, seq_len))
pae_labels = torch.randint(0, 64, (2, seq_len, seq_len))
pde_labels = torch.randint(0, 64, (2, seq_len, seq_len))
plddt_labels = torch.randint(0, 50, (2, seq_len))
resolved_labels = torch.randint(0, 2, (2, seq_len))

# Train

loss = alphafold3(
    num_recycling_steps = 2,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    is_molecule_types = is_molecule_types,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask,
    atom_pos = atom_pos,
    molecule_atom_indices = molecule_atom_indices,
    distance_labels = distance_labels,
    pae_labels = pae_labels,
    pde_labels = pde_labels,
    plddt_labels = plddt_labels,
    resolved_labels = resolved_labels
)

loss.backward()

# After much training ...

sampled_atom_pos = alphafold3(
    num_recycling_steps = 4,
    num_sample_steps = 16,
    atom_inputs = atom_inputs,
    atompair_inputs = atompair_inputs,
    molecule_ids = molecule_ids,
    molecule_atom_lens = molecule_atom_lens,
    additional_molecule_feats = additional_molecule_feats,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask
)

sampled_atom_pos.shape # (2, <atom_seqlen>, 3)
```

</details>

## Data preparation

<details>

### PDB dataset curation

To acquire the AlphaFold 3 PDB dataset, first download all complexes in the Protein Data Bank (PDB), and then preprocess them with the script referenced below. The PDB can be downloaded from the RCSB: https://www.wwpdb.org/ftp/pdb-ftp-sites#rcsbpdb. The Python script below (i.e., `pdb_mmcif_filtering.py`) assumes you have downloaded the PDB in the **mmCIF file format**, placing it at `data/pdb_data/unfiltered_mmcifs/`. On the RCSB website, navigate down to "Download Protocols", and follow the download instructions depending on your location.

For example, one can use the following command to download the PDB as a collection of mmCIF files:
```bash
rsync -rlpt -v -z --delete --port=33444 \
rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ ./data/pdb_data/unfiltered_mmcifs/
```

> WARNING: Downloading PDB can take up to 1TB of space.

After downloading, you should have a directory formatted like this:
https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/
```bash
00/
01/
02/
..
zz/
```

In this directory, unzip all the files:
```bash
find . -type f -name "*.gz" -exec gzip -d {} \;
```

Next run the commands
```bash
wget -P ./data/ccd_data/ https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
wget -P ./data/ccd_data/ https://files.wwpdb.org/pub/pdb/data/component-models/complete/chem_comp_model.cif.gz
```
from the project's root directory to download the latest version of the PDB's Chemical Component Dictionary (CCD) and its structural models. Extract each of these files using the following command:
```bash
find data/ccd_data/ -type f -name "*.gz" -exec gzip -d {} \;
```

### PDB dataset filtering

Then run the following with `pdb_dir`, `ccd_dir`, and `mmcif_output_dir` replaced with the locations of your local copies of the PDB, CCD, and your desired dataset output directory (i.e., `./data/pdb_data/unfiltered_mmcifs/`, `./data/ccd_data/`, and `./data/pdb_data/mmcifs/`).
```bash
python scripts/filter_pdb_mmcifs.py --mmcif_dir <pdb_dir> --ccd_dir <ccd_dir> --output_dir <mmcif_output_dir>
```

See the script for more options. Each mmCIF that successfully passes
all processing steps will be written to `mmcif_output_dir` within a subdirectory
named according to the mmCIF's second and third PDB ID characters (e.g. `5c`).

### PDB dataset clustering

Next, run the following with `mmcif_dir`, `ccd_dir`, and `clustering_output_dir` replaced, respectively, with your local output directory created using the dataset curation script above; with the location of your local CCD copy; and with your desired clustering output directory (i.e., `./data/pdb_data/mmcifs/`, `./data/ccd_data/`, and `./data/pdb_data/data_caches/clusterings/`):
```bash
python scripts/cluster_pdb_mmcifs.py --mmcif_dir <mmcif_dir> --ccd_dir <ccd_dir> --output_dir <clustering_output_dir>
```

See the script above for more options.

### PDB dataset caching

Now, run the following with `pdb_dir` and `cache_output_path` replaced with the location of your local copy of the PDB and your desired dataset cache output filepath (i.e., `./data/pdb_data/mmcifs/` and `./data/pdb_data/data_caches/chain_data_cache.json`).
```bash
python scripts/generate_mmcif_cache.py --mmcif_dir <pdb_dir> --output_path <cache_output_path>
```

See the script above for more options.

</details>

## Training

<details>

Train model with default configuration

```bash
# Train on CPU
python alphafold3_pytorch/train.py trainer=cpu

# Train on GPU
python alphafold3_pytorch/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# e.g., Train an initial set of weights
python alphafold3_pytorch/train.py experiment=alphafold3_initial_training.yaml
```

You can override any parameter from command line like this

```bash
python alphafold3_pytorch/train.py trainer.max_steps=1e6 data.batch_size=128
```

</details>

## Evaluation

<details>

Evaluate a trained set of weights on held-out test data

```bash
# e.g., Evaluate on GPU
python alphafold3_pytorch/eval.py trainer=gpu
```

</details>

## For developers

<details>

### Contributing

At the project root, run

```bash
bash contributing.sh
```

Then, add your module to `alphafold3_pytorch/models/components/alphafold3.py`, add your tests to `tests/test_alphafold3.py`, and submit a pull request. You can run the tests locally with

```bash
pytest tests/
```

### Dependency management

We use `pip` and `docker` to manage the project's underlying dependencies. Notably, to update the dependencies built by the project's `Dockerfile`, first edit the contents of the `dependencies` list in `pyproject.toml`, and then rebuild the project's `docker` image:

```bash
docker stop <container_id> # First stop any running `af3` container(s)
docker rm <container_id> # Then remove the container(s) - Caution: Make sure to push your local changes to GitHub before running this!
docker build -t af3 . # Rebuild the Docker image
docker run -v .:/data --gpus all -it af3 # # Lastly, (re)start the Docker container from the updated image
```

If you want to update the project's `pip` dependencies only, you can simply push to GitHub your changes to the `pyproject.toml` file.

### Code formatting

We use `pre-commit` to automatically format the project's code. To set up `pre-commit` (one time only) for automatic code linting and formatting upon each execution of `git commit`:

```bash
pre-commit install
```

To manually reformat all files in the project as desired:

```bash
pre-commit run -a
```

Refer to [pre-commit's documentation](https://pre-commit.com/) for more details.

</details>

## Citations

```bibtex
@article{Abramson2024-fj,
  title    = "Accurate structure prediction of biomolecular interactions with
              {AlphaFold} 3",
  author   = "Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans,
              Richard and Green, Tim and Pritzel, Alexander and Ronneberger,
              Olaf and Willmore, Lindsay and Ballard, Andrew J and Bambrick,
              Joshua and Bodenstein, Sebastian W and Evans, David A and Hung,
              Chia-Chun and O'Neill, Michael and Reiman, David and
              Tunyasuvunakool, Kathryn and Wu, Zachary and {\v Z}emgulyt{\.e},
              Akvil{\.e} and Arvaniti, Eirini and Beattie, Charles and
              Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and
              Congreve, Miles and Cowen-Rivers, Alexander I and Cowie, Andrew
              and Figurnov, Michael and Fuchs, Fabian B and Gladman, Hannah and
              Jain, Rishub and Khan, Yousuf A and Low, Caroline M R and Perlin,
              Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and
              Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine
              and Yakneen, Sergei and Zhong, Ellen D and Zielinski, Michal and
              {\v Z}{\'\i}dek, Augustin and Bapst, Victor and Kohli, Pushmeet
              and Jaderberg, Max and Hassabis, Demis and Jumper, John M",
  journal  = "Nature",
  month    = "May",
  year     =  2024
}
```

```bibtex
@inproceedings{Darcet2023VisionTN,
    title   = {Vision Transformers Need Registers},
    author  = {Timoth'ee Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:263134283}
}
```

```bibtex
@article{Arora2024SimpleLA,
    title   = {Simple linear attention language models balance the recall-throughput tradeoff},
    author  = {Simran Arora and Sabri Eyuboglu and Michael Zhang and Aman Timalsina and Silas Alberti and Dylan Zinsley and James Zou and Atri Rudra and Christopher R'e},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.18668},
    url     = {https://api.semanticscholar.org/CorpusID:268063190}
}
```

```bibtex
@article{Puny2021FrameAF,
    title   = {Frame Averaging for Invariant and Equivariant Network Design},
    author  = {Omri Puny and Matan Atzmon and Heli Ben-Hamu and Edward James Smith and Ishan Misra and Aditya Grover and Yaron Lipman},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2110.03336},
    url     = {https://api.semanticscholar.org/CorpusID:238419638}
}
```

```bibtex
@article{Duval2023FAENetFA,
    title   = {FAENet: Frame Averaging Equivariant GNN for Materials Modeling},
    author  = {Alexandre Duval and Victor Schmidt and Alex Hernandez Garcia and Santiago Miret and Fragkiskos D. Malliaros and Yoshua Bengio and David Rolnick},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2305.05577},
    url     = {https://api.semanticscholar.org/CorpusID:258564608}
}
```

```bibtex
@article{Wang2022DeepNetST,
    title   = {DeepNet: Scaling Transformers to 1, 000 Layers},
    author  = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Dongdong Zhang and Furu Wei},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2203.00555},
    url     = {https://api.semanticscholar.org/CorpusID:247187905}
}
```

```bibtex
@inproceedings{Ainslie2023CoLT5FL,
    title   = {CoLT5: Faster Long-Range Transformers with Conditional Computation},
    author  = {Joshua Ainslie and Tao Lei and Michiel de Jong and Santiago Ontan'on and Siddhartha Brahma and Yury Zemlyanskiy and David Uthus and Mandy Guo and James Lee-Thorp and Yi Tay and Yun-Hsuan Sung and Sumit Sanghai},
    year    = {2023}
}
```

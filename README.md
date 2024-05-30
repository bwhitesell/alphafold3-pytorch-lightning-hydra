<div align="center">

# AlphaFold 3 - PyTorch

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/s41586-024-07487-w) -->

<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

<img src="./img/alphafold3.png" width="600">

</div>

## Description

Implementation of <a href="https://www.nature.com/articles/s41586-024-07487-w">AlphaFold 3</a> in Pytorch

You can chat with other researchers about this work <a href="https://discord.gg/x6FuzQPQXY">here</a>

## Appreciation

- <a href="https://github.com/joseph-c-kim">Joseph</a> for contributing the Relative Positional Encoding and the Smooth LDDT Loss!

- <a href="https://github.com/engelberger">Felipe</a> for contributing Weighted Rigid Align, Express Coordinates In Frame, Compute Alignment Error, and Centre Random Augmentation modules!

- <a href="https://github.com/amorehead">Alex</a> for fixing various issues in the transcribed algorithms

- <a href="https://github.com/gitabtion">Heng</a> for pointing out inconsistencies with the paper and pull requesting the solutions

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

### Pip

```bash
pip install alphafold3-pytorch
```

### Docker

```bash
# Clone project
git clone https://github.com/lucidrains/alphafold3-pytorch
cd alphafold3-pytorch

# Build Docker container
docker build -t af3 .

# Run container (with GPUs)
docker run --gpus all -it af3
```

## Usage

Run a sample script

```bash
bash scripts/usage.py
```

which is based on the following sample code:

```python
import torch
from alphafold3_pytorch import AlphaFold3

alphafold3 = AlphaFold3(
    dim_atom_inputs = 77,
    dim_template_feats = 44
)

# Mock inputs

seq_len = 16
residue_atom_lens = torch.randint(1, 3, (2, seq_len))
atom_seq_len = residue_atom_lens.sum(dim = -1).amax()

atom_inputs = torch.randn(2, atom_seq_len, 77)
atompair_inputs = torch.randn(2, atom_seq_len, atom_seq_len, 5)
additional_residue_feats = torch.randn(2, seq_len, 10)

template_feats = torch.randn(2, 2, seq_len, seq_len, 44)
template_mask = torch.ones((2, 2)).bool()

msa = torch.randn(2, 7, seq_len, 64)
msa_mask = torch.ones((2, 7)).bool()

# Required for training, but omitted on inference

atom_pos = torch.randn(2, atom_seq_len, 3)
residue_atom_indices = residue_atom_lens - 1 # last atom, as an example

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
    residue_atom_lens = residue_atom_lens,
    additional_residue_feats = additional_residue_feats,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask,
    atom_pos = atom_pos,
    residue_atom_indices = residue_atom_indices,
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
    residue_atom_lens = residue_atom_lens,
    additional_residue_feats = additional_residue_feats,
    msa = msa,
    msa_mask = msa_mask,
    templates = template_feats,
    template_mask = template_mask
)

sampled_atom_pos.shape # (2, <atom_seqlen>, 3)
```

## Data preparation

To acquire the AlphaFold 3 PDB dataset, first download all complexes in the Protein Data Bank (PDB), and then preprocess them with the script referenced below. The PDB can be downloaded from the RCSB: https://www.wwpdb.org/ftp/pdb-ftp-sites#rcsbpdb. The script below assumes you have downloaded the PDB in the **mmCIF file format** (e.g., placing it at `data/mmCIF/`). On the RCSB website, navigate down to "Download Protocols", and follow the download instructions depending on your location.

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

Then run the following with <pdb_dir> replaced with the location of your local copy of the PDB.
```python
python notebooks/alphafold3_pdb_dataset_curation.py --mmcif_dir <pdb_dir>
```

See the script for more options. Each mmCIF will be written as a pickle file that
we read and process in the data loading pipeline. A `metadata.csv` will be saved
that contains the pickle path of each example as well as additional information
about each example for faster filtering.

## Training

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

## Evaluation

Evaluate a trained set of weights on held-out test data

```bash
# e.g., Evaluate on GPU
python alphafold3_pytorch/eval.py trainer=gpu
```

## For developers

### Contributing

At the project root, run

```bash
bash contribute.sh
```

Then, add your module to `alphafold3_pytorch/alphafold3.py`, add your tests to `tests/test_alphafold3.py`, and submit a pull request. You can run the tests locally with

```bash
pytest tests/
```

### Dependency management

We use `pip` and `docker` to manage the project's underlying dependencies. Notably, to update the dependencies built by the project's `Dockerfile`, first edit the contents of the `dependencies` list in `pyproject.toml`, and then rebuild the project's `docker` image:

```bash
docker stop <container_id> # First stop any running `af3` container(s)
docker rm <container_id> # Then remove the container(s) - Caution: Make sure to push your local changes to GitHub before running this!
docker build -t af3 . # Rebuild the Docker image
docker run --gpus all -it af3 # # Lastly, (re)start the Docker container from the updated image
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

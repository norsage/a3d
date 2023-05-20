# Antigen-Aware Antibody Design (A3D)
This repo contains code for training a T5 transformer on a SAbDab dataset to generate Fv-fragment of antibody given the linear epitope sequence of the target antigen.

### Create conda environment

```bash
conda env create -f environment.yaml
conda activate a3d
```

### Obtaining and preprocessing SAbDab dataset
Visit SAbDab [website](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?all=true#downloads)

Download an archive of all structures to `data/` directory and extract it.
You will also need summary tsv file, place it in `data/` as well.

NB: of course you can place data elsewhere, but in this case you'll need to adjust the arguments of the preprocessing script

Run preprocessing:
```python
python scripts/process_sabdab.py
```

### Training
Training arguments are configured with [Hydra](https://hydra.cc/), for details look into `conf/train.yaml`.

Run training script:
```bash
python scripts/train.py
```

Track training with [Aim](https://aimstack.readthedocs.io/en/latest/):
```bash
cd logs/a3d
aim up
```

### Inference
Refer to `inference.ipynb` how to run inference with a trained model

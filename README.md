# POTC: Prototype Prompting for Cross-Domain Recommendation

POTC learns correspondence between source and target user prototypes from overlapping user identities and optimal transport, then uses target-domain prototype prompts for cross-domain recommendation.

## Getting started

1. Clone the repository and install the dependencies (Python 3.10 or newer):

   ```bash
   git clone https://github.com/yspohoo/POTC.git
   cd POTC
   python -m pip install -r requirement.txt
   ```

2. Obtain the Amazon review JSON files or Douban movie/music review TSV files. Set their paths in `variable.py`. For Douban, also set `DOUBAN_PROCESSED_ROOT` to a new or empty output directory.

3. Start Jupyter and run `amazon-preprocess.ipynb` or `douban_preprocess.ipynb` to prepare the chosen dataset:

   ```bash
   jupyter notebook
   ```

4. Pretrain the source and target encoders for the chosen direction. For example:

   ```bash
   python pretrain.py --dataset amazon --source movie --target music
   ```

   For Douban, use `--dataset douban`. Select the CUDA device in `pretrain.py` to match your machine. Each task saves its encoders under its own `weights/` directory.

5. Open `main-train-edited.ipynb`, set `dataset`, `source_domain`, `target_domain`, the search ranges and GPU allocation, then run the notebook. Ray Tune selects the checkpoint by validation HR@10 and reports held-out test results. Use the same dataset and direction as in step 4.

## Training example

![Example Ray Tune training output](train-example.jpg)

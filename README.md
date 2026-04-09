# Summarization-Tiny-LM
The goal for this project is to create a modern encoder-decoder transformer model that can summarize text while being small enough to run on a consumer GPU with 8GB of vram or less. The model will be trained on [newsroom](https://arxiv.org/pdf/1804.11283), a dataset of reference text and summary pairs collected from various news sources. 

# Training
4 training runs were performed with diferent hyperparamaters and dataset filtering. The train and model files at the top level directory are the final and best versions. Training was tracked using mlflow. You can view all the runs with `mlflow server --backend-store-uri sqlite:///mlflow.db`

# Running

## Dataset
Download the Newsroom dataset from https://lil.nlp.cornell.edu/newsroom/download/index.html (requires registration). Extract `train.jsonl.gz`, `dev.jsonl.gz`, and `test.jsonl.gz` into a directory and point `DATA_PATH` in `data.ipynb` to it.

## Dependencies
```bash
pip install torch torchtune tokenizers emoji pandas matplotlib seaborn mlflow summaries
```

## Data Preprocessing
Run `data.ipynb` end to end. Update `DATA_PATH` to point to your local Newsroom data directory.

## Training
Update `DATA_FILE` in `train.py` to point to the preprocessed CSV, then launch with:
```bash
torchrun --nproc_per_node=2 train.py
```
Set `--nproc_per_node` to your number of GPUs. Checkpoints are saved to `./checkpoints/`.

# Summarization-Tiny-LM
The goal for this project is to create a modern encoder-decoder transformer model that can summarize text while being small enough to run on a consumer GPU with 8GB of vram or less. The model will be trained on [newsroom](https://arxiv.org/pdf/1804.11283), a dataset of reference text and summary pairs collected from various news sources. 

# Training
4 training runs were performed with diferent hyperparamaters and dataset filtering. The train and model files at the top level directory are the final and best versions. Training was tracked using mlflow. You can view all the runs with `mlflow server --backend-store-uri sqlite:///mlflow.db`

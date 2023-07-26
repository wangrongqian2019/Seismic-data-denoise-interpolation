Repository for the paper "**Adapting the residual dense network for seismic data denoising and upscaling**"

# Usage
This is a supervised learning approach, consisting of 3 steps:

## Step 1: Generating training data
You can generate training data in any way you prefer. We utilized the Python package [Devito](https://www.devitoproject.org/) to generate the training data.

## Step 2: Training
**train.py** is a framework for training process. Before using it, you need to make some modifications. First, you need to load your own training data in the **train.py**. Additionally, you can also adjust some hyperparameters.

## Step 3: Testing
**test.py** is a framework for testing process. You need to load your own testing data in the **test.py**.

# Description

# Citation
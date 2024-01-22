# mcdd_destroyer
## Introduction

A powerful AI to recognize mcdd and destroy it.  
Any mcdd will be destroyed by me!

## Setup

The environment can be set up with ``requirements.txt``. For example with conda:
```
conda create --name mcdd_destroyer python=3.9.17
conda activate mcdd_destroyer
pip install -r requirements.txt
```

## Training with  🤗 Accelerate

To train with default parameters and options:
```bash
cd mcdd
accelerate launch --config_file accelerate_config.yaml train.py --data-path your_data_path --results-path your_results_path
```

## Evaluating from checkpoint

```bash
cd mcdd
python eval.py --data-path your_data_path --results-path your_results_path
```

## Usage

Save the images you want to detect in the ``data`` folder(default).   
Then find the ``model.pt`` file in the ``results`` folder(default).It may be ``mcdd\results\test1\model.pt``,for example.  
When the model returns 1 means that the image contains mcdd, and 0 means that the image does not contain mcdd.
Then, run the following command:

```bash
cd mcdd
python main.py --model-path your_model_pt_path --data-path your_data_path
```

__Notice:__ We recommend ``.jpg`` format for the images(I have not tested other formats).And each time, only one image is allowed to be detected.

## Notes
The files in ``mcdd`` folder are used to build models. And the files outside ``mcdd`` folder are used to process the training data.Considering the size of the training data, I just uploaded a ``.zip`` file. You can unzip it and put it in the ``mcdd_dataset`` folder(default).
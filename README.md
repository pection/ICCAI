# LXMERT Experiments

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code and instructions for reproducing our experiments on LXMERT using the CLEVR-HOPE dataset.

## Reproducing our results

1. Checkout this repo, and it's submodule (vqa-framework)

```shell
REPO_DIR="/this/is/a/dir/"  # absolute path to the repo directory, ending in /
cd $REPO_DIR
git clone https://github.com/ikb-a/lxmert-systematicity-gap-in-vqa
```


2. Download LXMERT VQA pretrain to snap/pretrained/model_LXRT.pth To do so, from the root directory, run:

```shell
mkdir -p snap/pretrained
wget http://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```


3. Setup the environment:

```shell
# Create the python3 virtual env
python3 -m venv alt_env

# activate environment
source alt_env/bin/activate

# Install requirements, depending on python3 version
# For python 3.7, use vqa-framework/requirements.txt,
pip install -r vqa-framework/requirements.txt
# For python 3.10 + CUDA10.2, use the requirements.txt in the root,
# pip install -r requirements.txt
# For python 3.10 + CUDA11.5, use the alt_requirements.txt in the root,
# pip install -r alt_requirements.txt

#Add the vqa-framework to the virtual env by running
pip install -e ./vqa-framework
```

```shell
# Create a symlink to where the checkpoints will go (i.e., snap/clevr should end up being a symlink to CKPT_DIR)
CKPT_DIR="/checkpoints/path/" # where to store checkpoints, ending in /
cd snap
ln -s $CKPT_DIR clevr
```

4. Download the dataset of FRCNN features extracted from CLEVR-HOPE to the [./datasets](./datasets/) directory. The files are available for download from  the Internet Archive from [https://archive.org/details/CLEVR-HOPE_FRCNN](https://archive.org/details/CLEVR-HOPE_FRCNN). Note that you can download the dataset to a different location, so long as you edit the file [./src/clevr_tasks/symlink_config.yaml](./src/clevr_tasks/symlink_config.yaml)


5. To run our finetuning experiment run the `train_finetune.sh` script.

```shell
# Select the number of questions to finetune the model on. 
# We experimented with 25000, 200000, and 560000
MAX_Q=25000

# Select the index of the HOP (value must be between 0 and 28)
HOP_IDX=0

# Select which run of the experiment it is. We performed 3 runs.
RUN=1

# Run finetuning of LXMERT
./src/clevr_tasks/train_finetune.sh "$MAX_Q" "$HOP_IDX" "$RUN"
```

6. To evaluate the finetuning experiment, run the evaluation script:

```shell
# Select the number of questions to finetune the model on. 
# We experimented with 25000, 200000, and 560000
MAX_Q=25000

# Select the index of the HOP (value must be between 0 and 28)
HOP_IDX=0

# Select which run of the experiment it is. We performed 3 runs.
RUN=1

# Evaluate finetuned LXMERT
./src/clevr_tasks/test_finetune.sh "$MAX_Q" "$HOP_IDX" "$RUN"
```

7. To run our from-scratch experiment run the `train_scratch.sh` script.

```shell
# Select the number of questions to finetune the model on. 
# We experimented with 25000, 200000, and 560000
MAX_Q=25000

# Select the index of the HOP (value must be between 0 and 28)
HOP_IDX=0

# Select which run of the experiment it is. We performed 3 runs.
RUN=1

# Run finetuning of LXMERT
./src/clevr_tasks/train_scratch.sh "$MAX_Q" "$HOP_IDX" "$RUN"
```

8. To evaluate our from-scratch experiment, run the `test_scratch.sh` script.

```shell
# Select the number of questions to finetune the model on. 
# We experimented with 25000, 200000, and 560000
MAX_Q=25000

# Select the index of the HOP (value must be between 0 and 28)
HOP_IDX=0

# Select which run of the experiment it is. We performed 3 runs.
RUN=1

# Evaluate finetuned LXMERT
./src/clevr_tasks/test_scratch.sh "$MAX_Q" "$HOP_IDX" "$RUN"
```

## Reference

If you've used CLEVR-HOPE in your work, please do cite the paper and let us know by updating the [page for other works using CLEVR-HOPE](https://github.com/ikb-a/systematicity-gap-in-vqa/blob/main/FOLLOWUP.md)!

```bibtex
@inproceedings{berlot-attwell-etal-2024-attribute,
    title = "Attribute Diversity Determines the Systematicity Gap in {VQA}",
    author = "Berlot-Attwell, Ian  and
      Agrawal, Kumar Krishna  and
      Carrell, Annabelle Michael  and
      Sharma, Yash  and
      Saphra, Naomi",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.537",
    doi = "10.18653/v1/2024.emnlp-main.537",
    pages = "9576--9611",
    abstract = "Although modern neural networks often generalize to new combinations of familiar concepts, the conditions that enable such compositionality have long been an open question. In this work, we study the systematicity gap in visual question answering: the performance difference between reasoning on previously seen and unseen combinations of object attributes. To test, we introduce a novel diagnostic dataset, CLEVR-HOPE. We find that the systematicity gap is not reduced by increasing the quantity of training data, but is reduced by increasing the diversity of training data. In particular, our experiments suggest that the more distinct attribute type combinations are seen during training, the more systematic we can expect the resulting model to be.",
}
```

This codebase is modified from [the original LXMERT codebase](https://github.com/airsplay/lxmert).

# ICCAI

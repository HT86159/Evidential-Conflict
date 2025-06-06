# Visual-Hallucination-Detection-in-Large-Vision-Language-Models-via-Evidential-Conflict
When running inference code for LVLMs, it is necessary to modify the source code of the transformers to extract the probability, probability distribution, and lm_head_hidden_state weight of LVLMs when predicting each token, so as to calculate various subsequent uncertainty indicators.

## System Requipment
We here discuss hardware and software system requirements.

### Hardware Dependencies
Generally speaking, our experiments require modern computer hardware which is suited for usage with large language models (LLMs).

Requirements regarding the system's CPU and RAM size are relatively modest: any reasonably modern system should suffice, e.g. a system with an Intel 10th generation CPU and 16 GB of system memory or better.

More importantly, all our experiments make use of one or more Graphics Processor Units (GPUs) to speed up LLM inference. Without a GPU, it is not feasible to reproduce our results in a reasonable amount of time. The particular GPU necessary depends on the choice of LLM: LLMs with more parameters require GPUs with more memory. For smaller models (7B or 13B parameters), they require the use of one Nvidia RTX3090 GPU. Largest models with 34B parameters require the use of two Nvidia RTX3090 GPUs simultaneously.

One can reduce the precision to float16 or int8 to reduce memory requirements without significantly affecting model predictions and their accuracy. We use float16 for 70B models by default, and int8 mode can be enabled for any model by suffixing the model name with -int8.

### Software Dependencies
Our code relies on Python 3.11 with PyTorch 2.1.

Our systems run the Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-89-generic x86_64) operating system.

In environment_export.yaml we list the exact versions for all Python packages used in our experiments. We generally advise against trying to install from this exact export of our conda environment. Please see below for installation instructions.

Although we have not tested this, we would expect our code to be compatible with other operating systems, Python versions, and versions of the Python libraries that we use.

## Installation Guide
For the deployment of each model and the setup of the corresponding virtual environment, please follow the installation instructions published by the official Github repository of each model.

## Demo
At present, we have not yet developed a Demo.

## Further Instructions

### Repository Structure
This rspository is devided into five files, which are "infer", "infer_results", "measures", "models", "model_weights".
Among them, the "infer" folder stores the code for model inference. 
The "infer_results" folder stores the results generated by running the model inference files.
The "model_weights" folder stores the weights of the last hidden layer of each model.
The "models" folder stores the files required for the deployment of each model.
The "measures" folder contains code for processing model inference results, including code for verifying the correctness of model outputs using GPT-4o, code for calculating various uncertainty metrics, and code for computing evaluation metrics such as AUROC and ACC on the final results.

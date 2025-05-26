# Visual-Hallucination-Detection-in-Large-Vision-Language-Models-via-Evidential-Conflict
When running inference code for LVLMs, it is necessary to modify the source code of the transformers to extract the probability, probability distribution, and lm_head_hidden_state weight of LVLMs when predicting each token, so as to calculate various subsequent uncertainty indicators.

## System Requipment
We here discuss hardware and software system requirements.

### Hardware Dependencies
Generally speaking, our experiments require modern computer hardware which is suited for usage with large language models (LLMs).

Requirements regarding the system's CPU and RAM size are relatively modest: any reasonably modern system should suffice, e.g. a system with an Intel 10th generation CPU and 16 GB of system memory or better.

More importantly, all our experiments make use of one or more Graphics Processor Units (GPUs) to speed up LLM inference. Without a GPU, it is not feasible to reproduce our results in a reasonable amount of time. The particular GPU necessary depends on the choice of LLM: LLMs with more parameters require GPUs with more memory. For smaller models (7B parameters), desktop GPUs such as the Nvidia TitanRTX (24 GB) are sufficient. For larger models (13B), GPUs with more memory, such as the Nvidia A100 server GPU, are required. Our largest models with 34B parameters require the use of two Nvidia RTX3090 GPUs simultaneously.

One can reduce the precision to float16 or int8 to reduce memory requirements without significantly affecting model predictions and their accuracy. We use float16 for 70B models by default, and int8 mode can be enabled for any model by suffixing the model name with -int8.

### Software Dependencies

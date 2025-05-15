# visual-hallucination-detection-in-large-vision-language-models-via-evidential-conflict
When running inference code for LVLMs, it is necessary to modify the source code of the transformers to extract the probability, probability distribution, and lm_head_hidden_state weight of LVLMs when predicting each token, so as to calculate various subsequent uncertainty indicators.

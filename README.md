# Hierarchy-Aware T5 with Path-Adaptive Mask Mechanism for Hierarchical Text Classification

### PAMM-HIA-T5
PAMM-HiA-T5 consists of the Hierarchy-Aware T5 and the Path-Adaptive Mask Mechanism.
The project consists of following parts:
+ data: Data dir for the preprocessed RCV1, NYT, WOS datasets (because of the datasets' size exceeds the available max limit set by ARR, we only upload a representative subset of them). The original datasets could refer to [RCV1-V2](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm), [NYT](https://catalog.ldc.upenn.edu/LDC2008T19) and [WOS](https://github.com/kk7nc/HDLTex). 
+ pretrain_model: Download the relevant files of the pre-training T5 model including pytorch_model.bin, config.json, tokenizer.json, spiece.model, etc. from [T5-base](https://huggingface.co/t5-base/tree/main), and then put them in the project path: PAMM-HiA-T5/pretrain_model/t5-base.
+ utils.py: The data processing and data loader of PAMM-HiA-T5.
+ dmask.model_t5_4_classification & train_dmask.py: The main model of PAMM-HiA-T5 and its training script.
+ train.py: The main model of HiA-T5 and its training script.
+ test.py: The test script of PAMM-HiA-T5 or HiA-T5.

### Requirements
+ python 3.7.9
+ pytorch 1.7.0
+ transformers 2.9.0

### Train & Test
The hyperparameters of PAMM-HiA-T5 are configured in the `args_dict` of `train_dmask.py`. You can change all hyperparameters and run `train_dmask.py` to train PAMM-HiA-T5 on different settings. To test the model, you can change the `ckpt_path`, `dataset`, and `badcase_path` in `test.py` and then run `test.py`.

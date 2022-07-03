# Exploring Non-Autoregressive Text Style Transfer
This is the PyTorch implementation of the EMNLP2021 short paper "Exploring Non-Autoregressive Text Style Transfer"[[pdf](https://aclanthology.org/2021.emnlp-main.730.pdf)].

### Overview
- `src/` contains the implementations of the proposed method
```
src
|---train_classifier.py # train the cnn classifiers used in training/evaluation
|---train_lm.py # train the language model used in evaluation
|---train_st.py # train the autoregressive style transfer model
|---train_natst.py # train the non-autoregressive style transfer model
|---evaluator.py # evaluate the predictions of the style transfer model

```
- `outputs/` contains the predictions of the proposed non-autoregressive model enhanced by knowledge distillation and contrastive learning (i.e., BaseNAR+KD+CL)
```
outputs
|---GYAFC
|---|---natst.0 # formal->informal outputs of BaseNAR+KD+CL model
|---|---natst.1 # informal->formal outputs of BaseNAR+KD+CL model
|---|---test.0 # formal input
|---|---test.1 # informal input
|---yelp
|---|---natst.0 # negative->positive outputs of BaseNAR+KD+CL model
|---|---natst.1 # positive->negative outputs of BaseNAR+KD+CL model
|---|---test.0 # negative input
|---|---test.1 # positive input
```

### Dependencies
```
python == 3.7
pytorch == 1.3.1
```

### Citation
If you find our paper and code useful, please cite:
```
@inproceedings{ma-li-2021-exploring,
    title = "Exploring Non-Autoregressive Text Style Transfer",
    author = "Ma, Yun and Li, Qing",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "9267--9278",
}

```



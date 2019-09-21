# PyTorchTransformer

***
Code for a NMT task based on Transformer model and Mutil30kdataset in pytorch
***

Only contain the train and validate step but not test step
***

**Dataset**：Mutil30k de-en Translation Dataset
**Proformance**：Training accuracy：≈ 89.2%
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&nbsp;                 Validation accuary：≈62.0%
**Hardware**: 2*Nvidia Titan Xp
**Software**: Pytorch 1.1 CUDA 10.0 CuDnn 7.5
**Hyperparameters**
 Training with 150 epoch and all the same hyperparameters as the original model in the paper
### Reference

[Attention is all you need](https://arxiv.org/abs/1706.03762)
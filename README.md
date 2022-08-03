# NeRV: Neural Representations for Videos  (NeurIPS 2021)
### [Project Page](https://haochen-rye.github.io/NeRV) | [Paper](https://arxiv.org/abs/2110.13903) | [UVG Data](http://ultravideo.fi/#testsequences) 


[Hao Chen](https://haochen-rye.github.io),
Bo He,
Hanyu Wang,
Yixuan Ren,
Ser-Nam Lim],
[Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/)<br>
This is the official implementation of the paper "NeRV: Neural Representations for Videos ".

## Method overview
<img src="https://i.imgur.com/OTdHe6r.png" width="560"  />

## Get started
We run with Python 3.8, you can set up a conda environment with all dependencies like so:
```
pip install -r requirements.txt 
```

## High-Level structure
The code is organized as follows:
* [train_nerv.py](./train_nerv.py) includes a generic traiing routine.
* [model_nerv.py](./model_nerv.py) contains the dataloader and neural network architecure 
* [data/](./data) directory video/imae dataset, we provide big buck bunny here
* [checkpoints/](./checkpoints) directory contains some pre-trained model on big buck bunny dataset
* log files (tensorboard, txt, state_dict etc.) will be saved in output directory (specified by ```--outf```)

## Reproducing experiments

### Training experiments
The NeRV-S experiment on 'big buck bunny' can be reproduced with, NeRV-M and NeRV-L with ```9_16_58``` and ```9_16_112``` for ```fc_hw_dim``` respectively.
```
python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --act swish 
```

### Evaluation experiments
To evaluate pre-trained model, just add --eval_Only and specify model path with --weight, you can specify model quantization with ```--quant_bit [bit_lenght]```, yuo can test decoding speed with ```--eval_fps```, below we preovide sample commends for NeRV-S on bunny dataset
```
python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none  --act swish \
    --weight checkpoints/nerv_S.pth --eval_only 
```

### Decoding: Dump predictions with pre-trained model 
To dump predictions with pre-trained model, just add ```--dump_images``` besides ```--eval_Only``` and specify model path with ```--weight```
```
python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none  --act swish \
   --weight checkpoints/nerv_S.pth --eval_only  --dump_images
```

## Model Pruning

### Evaluate the pruned model
Prune a pre-trained model and fine-tune to recover its performance, with ```--prune_ratio``` to specify model parameter amount to be pruned, ```--weight``` to specify the pre-trained model, ```--not_resume_epoch``` to skip loading the pre-trained weights epoch to restart fine-tune
```
python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf prune_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
    --weight checkpoints/nerv_S.pth --not_resume_epoch --prune_ratio 0.4 
```

### Evaluate the pruned model
To evaluate pruned model, using ```--weight``` to specify the pruned model weight, ```--prune_ratio``` to initialize the ```weight_mask``` for checkpoint loading, ```eval_only``` for evaluation mode, ```--quant_bit``` to specify quantization bit length, ```--quant_axis``` to specify quantization axis
```
python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
    --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 0

```

### Distrotion-Compression result
The final bits-per-pixel (bpp) is computed by $$ModelParameter * (1 - ModelSparsity) * QuantBit / PixelNum$$.
We provide numerical results for distortion-compression (Figure 7, 8 and 11) at [psnr_bpp_results.csv](./checkpoints/psnr_bpp_results.csv) .

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{hao2021nerv,
    author = {Hao Chen, Bo He, Hanyu Wang, Yixuan Ren, Ser-Nam Lim, Abhinav Shrivastava },
    title = {NeRV: Neural Representations for Videos s},
    booktitle = {NeurIPS},
    year={2021}
}
```

## Contact
If you have any questions, please feel free to email the authors.

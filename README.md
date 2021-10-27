# NeRV: Neural Representations for Videos  (NeurIPS 2021)
### [Project Page](TODO) | [Paper](https://arxiv.org/abs/2110.13903) | [UVG Data](http://ultravideo.fi/#testsequences) 


[Hao Chen](https://haochen-rye.github.io),
[Bo He](),
[Hanyu Wang](),
[Yixuan Ren](),
[Ser-Nam Lim](),
[Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/)<br>
This is the official implementation of the paper "NeRV: Neural Representations for Videos ".


## Get started
You can set up a conda environment with all dependencies like so:
```
pip install -r requirements.txt 
```

## High-Level structure
The code is organized as follows:
* train_nerv.py includes a generic traiing routine.
* model_nerv.py contains the dataloader and neural network architecure 
* video/imae dataset in data directory, we provide big buck bunny here
* checkpoint directory contains some pre-trained model on big buck bunny dataset


## Reproducing experiments
The directory `experiment_scripts` contains one script per experiment in the paper.

### Training experiments
The NeRV-S experiment on 'big buck bunny' can be reproduced with
```
python train_nerv.py -e 300 --cycles 1  --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --act swish 
```

### Evaluation experiments
To evaluate pre-trained model, just add --eval_Only and specify model path with --weight, you can specify model quantization with ```--quant_bit [bit_lenght]```, yuo can test decoding speed with ```--eval_fps```, below we preovide sample commends for NeRV-S on bunny dataset
```
python train_nerv.py -e 300 --cycles 1  --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none  --act swish \
    --weight checkpoints/nerv_S.pth --eval_only 
```

### Dump predictions with pre-trained model 
To evaluate pre-trained model, just add --eval_Only and specify model path with ```--weight```
```
python train_nerv.py -e 300 --cycles 1  --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none  --act swish \
   --weight checkpoints/nerv_S.pth --eval_only  --dump_images
```


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
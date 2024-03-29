## Requirements
- easydict==1.9
- editdistance==0.5.3
- lmdb==1.2.1
- matplotlib==3.3.4
- opencv-python==4.5.2.52
- Pillow==8.2.0
- tensorboard==2.5.0
- tensorboard-data-server==0.6.1
- tensorboard-plugin-wit==1.8.0
- torch==1.2.0
- torchvision==0.2.1
- tqdm==4.61.0
- ipython

## Dataset 
Download all resources at [Dataset ](https://pan.baidu.com/s/1sWV2_DUFXk4YuF2E4aUqSQ) with password: 6dbe
* TextZoom dataset
* Pretrained weights of CRNN 
* Pretrained weights of Transformer-based recognizer

All the resources shoulded be placed under ```./dataset/mydata```, for example
```python
./dataset/mydata/train1
./dataset/mydata/train2
./dataset/mydata/test
./dataset/mydata/pretrain_transformer.pth
...
```

## Pre-trained Model
[[Model]](https://pan.baidu.com/s/1cIIppXNFwVYEJGyG8IK-zw) with password thle
```python
./checkpoint/train/epoch_best.pth

```


## Testing
```python
CUDA_VISIBLE_DEVICES=GPU_NUM python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus --resume YOUR_MODEL --test --test_data_dir ./dataset/mydata/test
```

## Training
Coming


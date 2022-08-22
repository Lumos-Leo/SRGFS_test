# Requirements
Pytorch conda environments, [torch_sr.yaml](torch_sr.yaml) for testing. You can use the following script to create conda environments.
```bash
conda env create -f torch_sr.yaml
```

# Pipeline

## 1. Prepare DIV2K Data
Download **benchmark** and put test datasets in data folder. Then the structure should look like:
> sr/datasets/benchmark
>> Set5
>>> HR
>>>> 0801.png

>>>> ...

>>>> 0900.png

>>> LR\_bicubic
>>>> X4
>>>>> 0801x4.png

>>>>> ...

>>>>> 0900x4.png

## 2. Testing
You can select three types of tasks (x2/X3/x4), just modify ```--scale``` and ```--test_model_path```.  

The first step is to activate the virtual environment. 
```bash
conda activate torch_sr
```
### 2.1 Benchmark Testing
Testing the benchmark (Set5/Set14/Urban100/Manga109/B100/DIV2K) performance, please run the following script.
```bash
python test.py --scale 4 --n_resblocks 8 --alpha 0.5 --test_model_path weights/SRGFS/X4/SRGFS-S_inference.pth --test_data_path sr/dataset/datasets --test_only --test_type benchmark
```
### 2.2 Single Image Testing
Testing single image, please run the following script.
```bash
python test.py --scale 4 --n_resblocks 8 --alpha 0.5 --test_model_path weights/SRGFS/X4/SRGFS-S_inference.pth --test_data_path sr/datasets/xxx.png --test_only --test_type single
```
### 2.3 Single File Testing
If all the images contain in one file, please run the following script.
```bash
python test.py --scale 4 --n_resblocks 8 --alpha 0.5 --test_model_path weights/SRGFS/X4/SRGFS-S_inference.pth --test_data_path sr/datasets/xxx.rgb --test_only --test_type big_data --super_resolution 2160x3840x3
```
**Note:**   
The order of ```--super_resolution``` is ```hxwxc```
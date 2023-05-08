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

## 3.Experimental Results
* Time cost was evaluated on Titan XP, and the size of input HR image was 1280x720.
### 3.1 X4 Result
|Model|Params|Macs|Time Cost|Set5|Set14|B100|Urban100|Manga109|DIV2K|  
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|  
|SESR'21|13.84K|0.79G|1.1ms|30.75/0.8714|27.62/0.7579|27.00/0.7166|24.61/0.7304|27.90/0.8644|29.52/0.8155|
|Ours_v1|14.80K|0.57G|15.2ms|31.08/0.8770|27.78/0.7632|27.11/0.7217|24.82/0.7416|28.33/0.8754|-/-|
|Ours_v2|8.96K|0.48G|5.3ms|31.02/0.8762|27.84/0.7623|27.08/0.7186|24.79/0.7387|28.27/0.8730|29.61/0.8174|

# Eye-Inpainting-for-Group-Photos

## Introduction
When taking group photos, it’s common for a perfectly lit, well-angled, and beautifully composed photo to be ruined by someone with their eyes closed, while other photos may have open eyes but fall short in overall quality. This issue is particularly frequent in large group settings, leading to repeated retakes that are time-consuming and stressful. To address this, we developed an efficient and natural correction technique that intelligently replaces closed eyes, preserving the ideal lighting and composition to create a perfect group photo where everyone looks their best, enhancing the shooting experience and ensuring every photo becomes a cherished memory.

## Installation
### Set Environment
```
git clone git@github.com:CEJiang/Eye-Inpainting-for-Group-Photos.git
conda install --yes --file conda_requirement.txt
pip install -r requirement.txt
```

### Set up GFPGAN
```
cd GFPGAN
python setup.py
```

## Usage
### clone this repo
```
git clone https://github.com/CEJiang/Eye-Inpainting-for-Group-Photos.git
```
### Download the CelebA-ID dataset
You can download CelebA-ID Benchmark dataset according to the [Dataset](https://github.com/bdol/exemplar_gans#celeb-id-benchmark-dataset)

### Train
```
python main.py --mode 0 --model_folder <Path to model folder> --train_folder <Path to the training data folder>
```
### Test
```
python main.py --mode 1 --i <Path to indentity image> --r <Path to reference image> --model_path <Path to model>
```


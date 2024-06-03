# PFCFuse: A Poolformer and CNN fusion network for Infrared-Visible Image Fusion
Poolformer-cnn图像融合框架
The implementation of our paper "PFCFuse: A Poolformer and CNN fusion network for Infrared-Visible Image Fusion".
## Recommended Environment:
python=3.8\
torch=1.12.1+cu113\
scipy=1.9.3\
scikit-image=0.19.2\
scikit-learn=1.1.3\
tqdm=4.62.0
## Network Architecture:
Our PFCFuse is implemented in ``net.py``.
## Training:
### Data preprocessing
Run 
```
python dataprocessing.py
```
### Model training
Run 
```
python train.py
```
## Testing:
Run 
```
python test_IVF.py
```

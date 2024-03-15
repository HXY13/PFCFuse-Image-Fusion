# PFCFuse
Codes for ***PFCFuse: A Poolformer and CNN fusion network for Infrared-Visible Image Fusion. ***

## 🌐 Usage

### ⚙ Network Architecture

Our PFCFuse is implemented in ``net.py``.

### 🏊 Training
**1. Virtual Environment**
```
# create virtual environment
conda create -n pfcfuse python=3.8.10
conda activate pfcfuse
# select pytorch version yourself
# install pfcfuse requirements
pip install -r requirements.txt
```

**2. Data Preparation**

Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it in the folder ``'./MSRS_train/'``.

**3. Pre-Processing**

Run 
```
python dataprocessing.py
``` 
and the processed training dataset is in ``'./data/MSRS_train_imgsize_128_stride_200.h5'``.

**4. PFCFuse Training**

Run 
```
python train.py
``` 
and the trained model is available in ``'./models/'``.

## 📖 Related Work

- Zhao Z, Bai H, Zhang J, et al. Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion[C]
//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

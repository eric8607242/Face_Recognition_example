# Face_Recognition_example

## Download Dataset
```bash
# Download training dataset
$ mkdir -p download/train
$ wget -O download/train/CASIA-WebFace.tar.gz https://www.dropbox.com/s/6x03igvbvfwx24w/CASIA-WebFace.tar.gz?dl=1

# Download testing dataset
$ mkdir -p download/test
$ wget -O download/test/CFP_FP.tar.gz https://www.dropbox.com/s/e3u7804yk54yqoj/CFP_FP.tar.gz?dl=1
$ wget -O download/test/LFW.tar.gz https://www.dropbox.com/s/d1y5o66dn8vcpvv/LFW.tar.gz?dl=1
```

## How to run
```bash
$ python3 main.py --config config/example.yml
```

### Eric Codebase
This is a example repo for face recognition. We cover the basic component for face recognition. (e.g., loss function, data loader, and inference)

**The main components in this repo.**
* Loss function
    * SphereFace
    * CosFace
    * ArgFace

**Actually, I am still confused about some implementation detail for face recognition. Hope anyone can help me to clarify more implementation detail.**

#### Train
* Softmax
```
python3 train.py --title [EXPERIMENT TITLE] --margin-module-name softmax
```
* SphereFace
```
python3 train.py --title [EXPERIMENT TITLE] --margin-module-name sphereface
```
* CosFace
```
python3 train.py --title [EXPERIMENT TITLE] --margin-module-name cosface
```
* ArcFace
```
python3 train.py --title [EXPERIMENT TITLE] --margin-module-name arcface
```

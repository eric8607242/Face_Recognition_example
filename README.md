# Face_Recognition_example

This is a example repo for face recognition. We cover the basic component for face recognition. (e.g., loss function, data loader, and inference)

**The main components in this repo.**
* Loss function
    * SphereFace
    * CosFace
    * ArgFace

**Actually, I am still confused about some implementation detail for face recognition. Hope anyone can help me to clarify more implementation detail.**

## Get Start
### Train
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

# Face Recognition Playground

Play with different face recognition techniques and observe the result.
![example.png](https://i.imgur.com/rqMOLG7.png)

## Download Dataset
```bash
# Download training dataset
$ mkdir -p download/train
$ wget -O download/train/CASIA-WebFace.tar.gz https://www.dropbox.com/s/6x03igvbvfwx24w/CASIA-WebFace.tar.gz?dl=1

# Download testing dataset
$ mkdir -p download/test
$ wget -O download/test/CFP_FP.tar.gz https://www.dropbox.com/s/e3u7804yk54yqoj/CFP_FP.tar.gz?dl=1
$ wget -O download/test/LFW.tar.gz https://www.dropbox.com/s/d1y5o66dn8vcpvv/LFW.tar.gz?dl=1

# Untar them by yourself
```

## How to run
```bash
# Train face recognition model with softmax
$ python3 main.py --config config/example.yml

# Train face recognition model with triplet loss
$ python3 main.py --config config/facenet.yml
```

## Comparision Result
|              | Face Verification Accuracy | Pretrained Model                                                   |
|:------------:|:--------------------------:|--------------------------------------------------------------------|
| Softmax Loss |            0.87            | [Link](https://www.dropbox.com/s/dulk91gxcb47hfa/example.pth?dl=1) |
| Triplet Loss |            0.90            | [Link](https://www.dropbox.com/s/gk0ybamj2zowreh/facenet.pth?dl=1) |

## Visualization of Embedding Space
- **Softmax CrossEntrop**  
![softmax-distribution](https://i.imgur.com/GqqQvUG.png)

- **Triplet Loss**  
![triplet-distribution](https://i.imgur.com/aTuXq4A.png)

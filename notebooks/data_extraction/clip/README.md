# CLIP
- Run [clipv2.ipynb](./clipv2.ipynb) to create clip feature for faiss index

## Input directory:

```
|- transnet
    |- keyframes
        |- L01_V001
            |- 001.jpg
            |- 002.jpg
            |- ...
        |- L01_V001_extra
            |- 000000.jpg
            |- 000002.jpg
            |- ...
        |- ...
```

## Output directory:

```
|- clip
    |- CLIPv2_features
        |- L01_V001.npy
        |- L01_V001_extra.npy
        |- ...
```
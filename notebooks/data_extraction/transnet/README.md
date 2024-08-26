# Keyframe extraction
- Run [transnetv2.ipynb](./transnetv2.ipynb) to extract shots from videos.
- Run [cutframe.ipynb](./cutframe.ipynb) to extract keyframes and keyframes metadata from shots.

## Input directory:
```
|- AIC_Video 
   |- Videos_L01
   |- Videos_L02
   |- ...
```

## Output directory:
```
|- SceneJSON 
   |- L01
      |- V001.json
      |- V002.json
      |- ...
   |- L02
   |- ...
|- Keyframes
   |- L01
      |- 000000.jpg
      |- ...
   |- L02
   |- ...
|- Keyframes_Metadata
   |- L01_V001.json
   |- L01_V002.json
   |- ...
```
# Sample output file
## SceneJSON
- Example `L01/V001.json`
```
[[0, 43], [44, 321], [322, 376], ...]
```
## Keyframes_Metadata
- Example `L01_V001.json`
```
{
   "L01_V001_000000": {
      "shot_index": 0,
      "frame_index": 0,
      "shot_start": 0,
      "shot_end": 43,
      "timestamp": 0.0,
      "video_path": "Videos_L01/video/L01_V001.mp4",
      "frame_path": "L01/V001/000000.jpg"
   },
   ...
}
```
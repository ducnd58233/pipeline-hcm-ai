# Keyframe extraction

- Run [transnetv2.ipynb](./transnetv2.ipynb) to extract shots from videos.
- Run [cutframe.ipynb](./cutframe.ipynb) to extract keyframes and keyframes metadata from shots.

**`Note`** files' name contains "extra", which is keyframes split from `transnetv2.ipynb` and `cutframe.ipynb`, and files' name without "extra" is **orgainizing commitee keyframes**.

## Input directory:

```
|- dataset
   |- AIC_Video
      |- Videos_L01
         |- video
            |- L01_V001.mp4
            |- L01_V002.mp4
            |- ...
      |- Videos_L02
      |- ...
```

## Output directory:

```
|- transnet
   |- scene_JSON
      |- L01
         |- V001.json
         |- V002.json
         |- ...
      |- L02
      |- ...
   |- keyframes
      |- L01_V001 (supplied by organizing commitee)
         |- 001.jpg (files' name is index in video csv file of map-keyframes directory)
         |- 002.jpg
         |- ...
      |- L01_V001_extra (split from transnetv2.ipynb and cutframe.ipynb)
         |- 000000.jpg (files' name is frame index in video)
         |- 000002.jpg
         |- ...
   |- keyframes_metadata
      |- L01_V001.json (being mapped from organizing commitee keyframe and map-keyframe directory)
      |- L01_V001_extra.json
      |- L01_V002.json
      |- L01_V002_extra.json
      |- ...
   |- map-keyframes (supplied from orgainizing commitee)
      |- L01_V001.csv
      |- L01_V002.csv
      |- ...
```

# Sample output file

## scene_JSON
This directory saves transition shots in each video

- Example file path `L01/V001.json`

```
[[0, 43], [44, 321], [322, 376], ...]
```

## keyframes_metadata

- Example `L01_V001_extra.json`

```
{
   "L01_V001_extra_000000": {
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

- Example `L01_V001.json`
```
{
   "L01_V001_000000": {
      "shot_index": 0,
      "frame_index": 0,
      "shot_start": 0,
      "shot_end": 5,
      "timestamp": 0.0,
      "video_path": "Videos_L01/video/L01_V001.mp4",
      "frame_path": "keyframes/L01_V001/001.jpg"
  },
  ...
}
```
## map-keyframes
This directory was supplied by organizing commitee

- Example `L01_V001.csv`

```
n,pts_time,fps,frame_idx
1,0.0,25.0,0
2,5.0,25.0,125
3,12.0,25.0,300
...
```
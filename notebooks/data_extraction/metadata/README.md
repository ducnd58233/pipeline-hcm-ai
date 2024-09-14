# Image multi-tags detection
- Run [tag.ipynb](./tag.ipynb) to detect image tags

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
|- metadata
    |- tag
        |- L01_V001.json
        |- L01_V001_extra.json
        |- ...
```

## Sample output file

### Multi-tag

- Example `tag/L01_V001.json`

```
{
  "001.jpg": [
        "city",
        "city skyline",
        "city view",
        "night",
        "night view",
        "sea",
        "sky",
        "skyline",
        "sun",
        "sunset",
        "water"
  ],
  "002.jpg": [
        "chair",
        "person",
        "interview",
        "man",
        "news",
        "screen",
        "sit",
        "stand",
        "stool",
        "television",
        "video",
        "woman"
  ],
  ...
}
```

- Example `tag/L01_V001_extra.json`

```
{
    "000000.jpg": [
        "city",
        "city skyline",
        "city view",
        "night",
        "night view",
        "sea",
        "sky",
        "skyline",
        "sun",
        "sunset",
        "water"
  ],
  "000002.jpg": [
        "city",
        "city skyline",
        "city view",
        "night",
        "night view",
        "sea",
        "sky",
        "skyline",
        "sun",
        "sunset",
        "water"
  ],
  ...
}
```

# Object detection
- Run [object_extraction.ipynb](./object_extraction.ipynb) to detect object detection

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
|- metadata
    |- object_extraction
        |- object_counts
            |- L01_V001
                |- 002_counts.json
                |- 003_counts.json
                |- ...
            |- L01_V001_extra
                |- 000006_counts.json
                |- 000229_counts.json
                |- ...
            |- ...
        |- object_detection
            |- L01_V001
                |- 002_detection.json
                |- 003_detection.json
                |- ...
            |- L01_V001_extra
                |- 000006_detection.json
                |- 000229_detection.json
                |- ...
            |- ...
```

## Sample output file

### Object count

- Example `object_extraction/object_counts/L01_V001/002_counts.json`

```
{"tie": 1, "person": 2}
```

- Example `object_extraction/object_counts/L01_V001/000006_counts.json`

```
{"stop sign": 1}
```

### Object detection

- Example `object_extraction/object_counts/L01_V001/002_detection.json`

```
[
  {
    "label": "tie",
    "score": 0.8844091296195984,
    "box": [
        541.9992065429688, 239.13400268554688, 564.4471435546875, 346.50616455078125
    ]
  },
  {
    "label": "person",
    "score": 0.8007513880729675,
    "box": [
        718.42431640625, 158.71981811523438, 855.2699584960938, 358.2477722167969
    ]
  },
  {
    "label": "person",
    "score": 0.7231229543685913,
    "box": [
        473.8626708984375, 155.57489013671875, 622.6842041015625, 617.9702758789062
    ]
  }
]
```

- Example `object_extraction/object_counts/L01_V001/000006_detection.json`

```
[
  {
    "label": "stop sign",
    "score": 0.7647601962089539,
    "box": [
        338.0318908691406, 253.27496337890625, 450.6507568359375, 450.9660339355469
    ]
  }
]
```
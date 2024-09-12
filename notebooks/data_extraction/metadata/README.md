# Image tags detection
- Run [tag.ipynb](./tag.ipynb) to detect image tags

## Output directory:
```
|- tag_output
   |- L01
      |- V001_tagging.json
      |- V002_tagging.json
      |- ...
   |- L02
   |- ...
```
# Sample output file

## Keyframes_Metadata
- Example `tag_output/L01/V001_tagging.json`

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
    "000001.jpg": [
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
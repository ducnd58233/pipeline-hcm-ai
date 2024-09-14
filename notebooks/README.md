<h1><center>Pipeline HCM AI CHALLENGE <br> Notebook</center></h1>

# Setup

- If running local

**Step 0** Download dataset

**Step 1** Move dataset to `data_extraction/dataset` folder

**Step 2** Rename dataset folder to `AIC_Video`

**Step 3** Open `pipeline.ipynb` and modified pipeline (optional) to run

**`Note`** If there is any unnecessaries step, comment it out for pipeline to not to run it

**`Note`** files' name contains "extra", which is keyframes split from `transnetv2.ipynb` and `cutframe.ipynb`, and files' name without "extra" is **orgainizing commitee keyframes**.

# Folder structures

```
|- data_extraction
    |- audio
        |- Audio
        |- audio_recognition
        |- audio_extraction_detection.ipynb
        |- audio_recognition_translation.ipynb
    |- clip
        |- CLIPv2_features
            |- L01_V001_extra.npy
            |- L01_V001.npy
            |- ...
        |- clipv2.ipynb
    |- dataset
        |- AIC_Video
            |- Videos_L01
                |- video
                    |- L01_V001.mp4
                    |- L01_V002.mp4
                    |- ...
            |- Videos_...
    |- metadata
        |- object_extraction
            |- object_counts
                |- L01_V001
                |- L01_V001_extra
                |- ...
            |- object_detection
                |- L01_V001
                |- L01_V001_extra
                |- ...
        |- tag
            |- L01_V001_extra.json
            |- L01_V001.json
            |- ...
        |- ocr
            |- L01_V001_extra.json
            |- L01_V001.json
            |- ...
        |- recognize-anything (download as running tag.ipynb)
        |- pretrained (download as running tag.ipynb)
        |- easyocr.ipynb
        |- object_extraction.ipynb
        |- rename_json_metadata.ipynb (optional for changing file structure)
        |- tag.ipynb
    |- transnet
        |- keyframes
            |- L01_V001
            |- L01_V001_extra
            |- ...
        |- keyframes_metadata
            |- L01_V001_extra.json
            |- L01_V001.json
            |- ...
        |- scene_JSON
            |- L01
                |- V001.json
                |- V002.json
                |- ...
            |- L02
                |- V001.json
                |- V002.json
                |- ...
            |- ...
        |- map-keyframes (using for mapping name of organizing commitee keyframe image with its information)
            |- L01_V001.csv
            |- L01_V002.csv
            |- ...
        |- TransNetV2
        |- rename_json_keyframe_metadata.ipynb (optional for changing file structure)
        |- transnetv2.ipynb
        |- cutframe.ipynb
|- indexing
    |- metadata_encoded
        |- object_detection
    |- faiss_clipv2_cosine_cpu.bin
    |- faiss_clipv2_cosine_gpu.bin (optional for choosing type of faiss)
|- keyframes_metadata.json
```

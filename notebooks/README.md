<h1><center>Pipeline HCM AI CHALLENGE <br> Notebook</center></h1>

# Setup

- If running local

**Step 0** Download dataset

**Step 1** Move dataset to `data_extraction/dataset` folder

**Step 2** Rename dataset folder to `AIC_Video`

**Step 3** Open `pipeline.ipynb` and modified pipeline (optional) to run

**`Note`** If there is any unnecessaries step, comment it out for pipeline to not to run it

# Folder structures
```
|- data_extraction
    |- audio
    |- clip
    |- dataset
        |- AIC_Video
            |- Videos_L01
                |- video
                    |- L01_V001.mp4
                    |- L01_V002.mp4
                    |- ...
            |- Videos_...
    |- metadata
    |- transnet
        |- Keyframes
        |- Keyframes_Metadata
        |- SceneJSON
        |- TransNetV2
|- indexing
    |- faiss_clipv2_cosine_cpu.bin
    |- faiss_clipv2_cosine_gpu.bin
|- keyframes_metadata.json
```
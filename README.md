<h1><center>Pipeline HCM AI CHALLENGE <br> Event Retrieval from Visual Data</center></h1>

## Setup 
Create Conda:
```
conda create -n hcm-ai python=3.11 -y
conda activate hcm-ai
conda install ffmpeg
```
Create venv
```sh
python -m venv venv
pre-commit install
```

Windows venv activation
```powershell
# In cmd.exe
venv\Scripts\activate.bat
# In PowerShell
venv\Scripts\Activate.ps1
```

Linux and MacOS venv activation
```sh
source venv/bin/activate
```
## Installation
```
pip install git+https://github.com/openai/CLIP.git
pip install --default-timeout=1000 -r requirements.txt
```

## Run 
```
python app.py
```

URL: http://0.0.0.0:5001/home?index=0



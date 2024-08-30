<h1><center>Pipeline HCM AI CHALLENGE <br> Event Retrieval from Visual Data</center></h1>

## Setup 
**Step 0.** (optional) Download Miniconda
```shell
# The version of Anaconda may be different depending on when you are installing`
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
# and follow the prompts. The defaults are generally good.`
```

**Step 1.** Create Conda (for python version):
```
conda create -n hcm-ai python=3.9 -y
conda activate hcm-ai
conda install ffmpeg
```
Create venv
```sh
python -m venv venv
pre-commit install
```

**Step 2.** Create venv (for python installer requirements)
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
cp .env.example .env
python run.py
```

URL: http://127.0.0.1:5001/

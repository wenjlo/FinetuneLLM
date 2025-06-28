# FinetuneLLM
Use unsloth fine tune llm

according to the [official documents](https://github.com/unslothai/unsloth/blob/main/README.md)  below
````
conda create --name finetune_env \
    python=3.11.10 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate finetune_env
````
❌ You will find some errors...


✅ It can be fixed by install these correct version .

(Here's a example for cuda-version = 12.6)
````
pip install --upgrade pip
pip install -U --pre xformers --index-url https://download.pytorch.org/whl/cu126
pip install "unsloth[cu126-torch270] @ git+https://github.com/unslothai/unsloth.git" bitsandbytes xformers
````
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.5.1 torchvision==0.15.2 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install transformers[torch]==4.46.3
pip install datasets==3.1.0
pip install accelerate[torch]==1.1.1
pip install deepspeed[torch]==0.16.0
pip install wandb==0.18.7
pip install huggingface_hub==0.26.3
pip install lmdb==1.5.1
pip install scikit-learn==1.6.1
pip install matplotlib
# pip install -U flash-attn --no-build-isolation
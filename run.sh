git config --global credential.helper store
mkdir /root/.huggingface/
mkdir /root/.kaggle
cp hf_token /root/.huggingface/token
cp kaggle.json /root/.kaggle/kaggle.jsons
pip install albumentations
pip install transformers
pip install kaggle
pip install wandb
#pip install librosa
#pip install phonemizer
pip install peft
pip install --upgrade transformers
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
#sudo apt-get install espeak

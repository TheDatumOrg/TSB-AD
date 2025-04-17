apt-get update
apt-get install -y wget unzip
wget https://www.thedatum.org/datasets/TSB-AD-U.zip
unzip -o TSB-AD-U.zip -d Datasets
wget https://www.thedatum.org/datasets/TSB-AD-M.zip
unzip -o TSB-AD-M.zip -d Datasets
pip install -r requirements.txt

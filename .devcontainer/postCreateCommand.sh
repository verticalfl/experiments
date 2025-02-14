export DEBIAN_FRONTEND="noninteractive"

apt update
apt upgrade -y
apt install -y sudo git python3 python3-pip python3-venv


python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
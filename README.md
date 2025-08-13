# samsung-qreport-rag

```
apt-get install git
git config --global user.email "jeongho.jang0404@gmail.com"
git config --global user.name "JeonghoJang"

mkdir my-volume
cd my-volume

git clone https://github.com/Je0ngh0/samsung-qreport-rag.git
cd samsung-qreport-rag

apt install python3.12-venv -y
python3 -m venv venv
source venv/bin/activate

pip install jupyter ipywidgets
pip install notebook
pip install langchain
pip install langchain-community
pip install langchain-experimental
pip install -qU langchain-huggingface
pip install sentence-transformers

touch .env
```

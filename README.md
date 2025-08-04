# Project Setup and Dependencies

## Required Python Packages

Please install the following packages to run the project:


absl-py==2.0.0
alabaster==0.7.13
astor==0.8.1
astunparse==1.6.3
autograd==1.6.2
autograd-gamma==0.5.0
ax-platform==0.2.6
Babel==2.14.0
botorch==0.6.6
cachetools==5.3.2
certifi==2023.11.17
charset-normalizer==3.3.2
cloudpickle==2.2.1
colorama==0.4.6
cycler==0.11.0
Cython==0.29.28
decorator==5.1.1
# Editable install with no version control (DeepPurpose==0.1.5)
# Editable install with no version control (descriptastorus==2.5.1)
dgllife==0.3.2
docutils==0.19
einops==0.6.1
exceptiongroup==1.2.0
filelock==3.12.2
flatbuffers==23.5.26
fonttools==4.38.0
formulaic==0.3.4
fsspec==2023.1.0
future==0.18.3
gast==0.3.3
gensim==4.2.0
google-auth==2.26.2
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
gpytorch==1.8.1
grpcio==1.60.0
h5py==2.10.0
huggingface-hub==0.16.4
hyperopt==0.2.7
idna==3.6
imagesize==1.4.1
importlib-metadata==6.7.0
iniconfig==2.0.0
install==1.3.5
interface-meta==1.3.0
Jinja2==3.1.3
joblib==1.3.2
Keras==2.3.1
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
kiwisolver==1.4.5
libclang==16.0.6
lifelines==0.27.8
littleutils==0.2.2
local-attention==1.9.0
Markdown==3.4.4
MarkupSafe==2.1.3
matplotlib==3.5.2
mock==5.1.0
multipledispatch==1.0.0
networkx==2.3
node2vec==0.4.3
numpy==1.18.5
oauthlib==3.2.2
ogb-lite==0.0.3
opt-einsum==3.3.0
outdated==0.2.2
packaging==23.2
pandas==1.3.2
pandas-flavor==0.6.0
Pillow==9.5.0
plotly==5.18.0
pluggy==1.2.0
prettytable==3.7.0
protobuf==3.19.6
psutil==5.9.8
py4j==0.10.9.7
pyasn1==0.5.1
pyasn1-modules==0.3.0
Pygments==2.17.2
pyparsing==3.1.1
pyro-api==0.1.2
pyro-ppl==1.8.6
pytest==7.4.4
python-dateutil==2.8.2
pytz==2023.3.post1
PyYAML==6.0.1
qc-procrustes==1.0.0
rdkit==2023.3.2
rdkit-pypi==2022.9.5
regex==2023.12.25
requests==2.31.0
requests-oauthlib==1.3.1
rsa==4.9
safetensors==0.4.2
scikit-learn==1.0.2
scipy==1.7.3
six==1.16.0
smart-open==6.4.0
snowballstemmer==2.2.0
Sphinx==5.3.0
sphinxcontrib-applehelp==1.0.2
sphinxcontrib-devhelp==1.0.2
sphinxcontrib-htmlhelp==2.0.0
sphinxcontrib-jsmath==1.0.1
sphinxcontrib-qthelp==1.0.3
sphinxcontrib-serializinghtml==1.1.5
subword-nmt==0.3.8
tenacity==8.2.3
tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.3.0
tensorflow-estimator==2.3.0
tensorflow-intel==2.11.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor==2.3.0
tf-geometric==0.1.6
tf-sparse==0.0.17
threadpoolctl==3.1.0
tokenizers==0.13.3
tomli==2.0.1
torch==1.13.1
torch-cluster==1.6.1+pt113cpu
torch-geometric==2.3.1
torch-scatter==2.1.1+pt113cpu
torch-sparse==0.6.17+pt113cpu
torch-spline-conv==1.2.2+pt113cpu
tqdm==4.66.2
transformers==4.30.2
typeguard==2.13.3
typing_extensions==4.7.1
urllib3==1.26.0
wcwidth==0.2.13
Werkzeug==2.2.3
wget==3.2
wrapt==1.16.0
xarray==0.20.2
zipp==3.15.0
texttable==1.7.0
pytdc==0.4.1
loguru==0.7.2
> Note:  
> The packages `DeepPurpose==0.1.5` and `descriptastorus==2.5.1` are expected to be installed via given source code.

---

## Reproducing Results

To reproduce the experimental results:

```bash
python Main.py

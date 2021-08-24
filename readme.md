# Experiments accompanying Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning

by Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer and Frank Hutter
arXiv:2007.04074v2 [cs.LG]
https://arxiv.org/abs/2007.04074

**This repository is provided as-is and we do not provide any maintenance. 
Necessary scripts will be ported to the main Auto-sklearn repository.**

## Code organization

This repository contains experiment scripts for the above mentioned paper submission. We
will clean up the code over time to make it easier to understand and reproduce our results
while keeping its functionality. The code is organized into two directories:

* *experiment_scripts* contains all scripts we used for the experiments. A manual
  describing in which order to execute the scripts is given in HowTo.md.
* *notebooks* contains several jupyter-notebooks which we used to analyze the results
  and produce the tables and plot given in the paper.

## Installation and requirements

We used the following script for installation:

```
MINICONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
wget $MINICONDA_URL -O miniconda.sh
bash miniconda.sh -b -p miniconda
source "miniconda/bin/activate"
conda create -n autosklearn2metadata -y
conda activate autosklearn2metadata
conda install gxx_linux-64 gcc_linux-64 swig python=3.8 ipython -y
which conda
which python
pip install openml
git clone https://github.com/automl/2020_IEEE_Autosklearn_experiments
git clone https://github.com/automl/auto-sklearn
cd auto-sklearn
git checkout 0.12.6
cat requirements.txt | xargs -n 1 -L 1 pip3 install
pip install scipy<1.7.0
pip install .
pip install git+https://github.com/automl/smac3@84960ee5978313527212ff49ba42984dc988f032
pip install lockfile ipython jupyter
conda deactivate
```

which resulted in the following output

```
_libgcc_mutex             0.1                        main
_openmp_mutex             4.5                       1_gnu
_sysroot_linux-64_curr_repodata_hack 3                   haa98f57_10
argon2-cffi               20.1.0                   pypi_0    pypi
async-generator           1.10                     pypi_0    pypi
attrs                     21.2.0                   pypi_0    pypi
auto-sklearn              0.12.6                   pypi_0    pypi
backcall                  0.2.0              pyhd3eb1b0_0
binutils_impl_linux-64    2.35.1               h27ae35d_9
binutils_linux-64         2.35.1              h454624a_30
bleach                    3.3.0                    pypi_0    pypi
ca-certificates           2021.5.25            h06a4308_1
certifi                   2021.5.30        py38h06a4308_0
cffi                      1.14.5                   pypi_0    pypi
chardet                   4.0.0                    pypi_0    pypi
click                     8.0.1                    pypi_0    pypi
cloudpickle               1.6.0                    pypi_0    pypi
configspace               0.4.19                   pypi_0    pypi
cython                    0.29.23                  pypi_0    pypi
dask                      2021.6.2                 pypi_0    pypi
debugpy                   1.3.0                    pypi_0    pypi
decorator                 5.0.9              pyhd3eb1b0_0        
defusedxml                0.7.1                    pypi_0    pypi
distributed               2021.6.2                 pypi_0    pypi
entrypoints               0.3                      pypi_0    pypi
fsspec                    2021.6.1                 pypi_0    pypi
gcc_impl_linux-64         9.3.0               h6df7d76_17        
gcc_linux-64              9.3.0               h1ee779e_30        
gxx_impl_linux-64         9.3.0               hbdd7822_17        
gxx_linux-64              9.3.0               h7e70986_30
heapdict                  1.0.1                    pypi_0    pypi
idna                      2.10                     pypi_0    pypi
ipykernel                 6.0.1                    pypi_0    pypi
ipython                   7.25.0                   pypi_0    pypi
ipython_genutils          0.2.0              pyhd3eb1b0_1        
ipywidgets                7.6.3                    pypi_0    pypi
jedi                      0.17.0                   py38_0        
jinja2                    3.0.1                    pypi_0    pypi
joblib                    1.0.1                    pypi_0    pypi
jsonschema                3.2.0                    pypi_0    pypi
jupyter                   1.0.0                    pypi_0    pypi
jupyter-client            6.1.12                   pypi_0    pypi
jupyter-console           6.4.0                    pypi_0    pypi
jupyter-core              4.7.1                    pypi_0    pypi
jupyterlab-pygments       0.1.2                    pypi_0    pypi
jupyterlab-widgets        1.0.0                    pypi_0    pypi
kernel-headers_linux-64   3.10.0              h57e8cba_10        
lazy-import               0.2.2                    pypi_0    pypi
ld_impl_linux-64          2.35.1               h7274673_9        
liac-arff                 2.5.0                    pypi_0    pypi
libffi                    3.3                  he6710b0_2        
libgcc-devel_linux-64     9.3.0               hb95220a_17        
libgcc-ng                 9.3.0               h5101ec6_17        
libgomp                   9.3.0               h5101ec6_17        
libstdcxx-devel_linux-64  9.3.0               hf0c5c8d_17        
libstdcxx-ng              9.3.0               hd4cf53a_17        
locket                    0.2.1                    pypi_0    pypi
lockfile                  0.12.2                   pypi_0    pypi
markupsafe                2.0.1                    pypi_0    pypi
matplotlib-inline         0.1.2                    pypi_0    pypi
minio                     7.0.4                    pypi_0    pypi
mistune                   0.8.4                    pypi_0    pypi
msgpack                   1.0.2                    pypi_0    pypi
nbclient                  0.5.3                    pypi_0    pypi
nbconvert                 6.1.0                    pypi_0    pypi
nbformat                  5.1.3                    pypi_0    pypi
ncurses                   6.2                  he6710b0_1        
nest-asyncio              1.5.1                    pypi_0    pypi
notebook                  6.4.0                    pypi_0    pypi
numpy                     1.21.0                   pypi_0    pypi
openml                    0.12.2                   pypi_0    pypi
openssl                   1.1.1k               h27cfd23_0
packaging                 21.0                     pypi_0    pypi
pandas                    1.2.5                    pypi_0    pypi
pandocfilters             1.4.3                    pypi_0    pypi
parso                     0.8.2              pyhd3eb1b0_0        
partd                     1.2.0                    pypi_0    pypi
pcre                      8.45                 h295c915_0        
pexpect                   4.8.0              pyhd3eb1b0_3        
pickleshare               0.7.5           pyhd3eb1b0_1003        
pip                       21.1.3           py38h06a4308_0        
prometheus-client         0.11.0                   pypi_0    pypi
prompt-toolkit            3.0.17             pyh06a4308_0        
psutil                    5.8.0                    pypi_0    pypi
ptyprocess                0.7.0              pyhd3eb1b0_2        
pyarrow                   4.0.1                    pypi_0    pypi
pycparser                 2.20                     pypi_0    pypi
pygments                  2.9.0              pyhd3eb1b0_0        
pynisher                  0.6.4                    pypi_0    pypi
pyparsing                 2.4.7                    pypi_0    pypi
pyrfr                     0.8.2                    pypi_0    pypi
pyrsistent                0.18.0                   pypi_0    pypi
python                    3.8.10               h12debd9_8        
python-dateutil           2.8.1                    pypi_0    pypi
pytz                      2021.1                   pypi_0    pypi
pyyaml                    5.4.1                    pypi_0    pypi
pyzmq                     22.1.0                   pypi_0    pypi
qtconsole                 5.1.1                    pypi_0    pypi
qtpy                      1.9.0                    pypi_0    pypi
readline                  8.1                  h27cfd23_0        
requests                  2.25.1                   pypi_0    pypi
scikit-learn              0.24.2                   pypi_0    pypi
scipy                     1.6.3                    pypi_0    pypi
send2trash                1.7.1                    pypi_0    pypi
setuptools                52.0.0           py38h06a4308_0
six                       1.16.0                   pypi_0    pypi
smac                      0.13.1                   pypi_0    pypi
sortedcontainers          2.4.0                    pypi_0    pypi
sqlite                    3.36.0               hc218d9a_0
swig                      4.0.2                h2531618_3        
sysroot_linux-64          2.17                h57e8cba_10            
tblib                     1.7.0                    pypi_0    pypi
terminado                 0.10.1                   pypi_0    pypi
testpath                  0.5.0                    pypi_0    pypi
threadpoolctl             2.1.0                    pypi_0    pypi
tk                        8.6.10               hbc83047_0        
toolz                     0.11.1                   pypi_0    pypi
tornado                   6.1                      pypi_0    pypi
traitlets                 5.0.5              pyhd3eb1b0_0        
urllib3                   1.26.6                   pypi_0    pypi
wcwidth                   0.2.5                      py_0        
webencodings              0.5.1                    pypi_0    pypi
wheel                     0.36.2             pyhd3eb1b0_0        
widgetsnbextension        3.5.1                    pypi_0    pypi
xmltodict                 0.12.0                   pypi_0    pypi
xz                        5.2.5                h7b6447c_0        
zict                      2.0.0                    pypi_0    pypi
zlib                      1.2.11               h7b6447c_3 
```

## Experimental results

All results are available under https://bwsyncandshare.kit.edu/s/y49gQfcDXBD7PSw

## Further notes

Running the code for the 1 hour setting will cost roughly one to two CPU years and should be done on a high-performance compute cluster.

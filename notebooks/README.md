# Excelsior-hackathEOn

## Getting Started

To get started from scratch, please follow the steps:

1. Check that you have `python` installed in your system. Open a terminal (*power shell* in Windows), and then type:

```
python --version
```

You should see something like `Python 3.6.9`. This tutorial was tested on `python == 3.10.12`. If you don't have python installed, visit [python.org](https://www.python.org/downloads/), and follow the instructions.

2. [Recommended] Create a virtual environment `virutalenv --python=python3 venv`, and activate it `source venv/local/bin/activate`.

3. Install *Jupyterlab* `pip install jupyterlab`.

4. Install all the required packages, e.g., by executing `pip install -r requirements`. 

5. Download the data from this [link](https://syncandshare.lrz.de/getlink/fiFXcnxNYWaxALLmRzAq6W/dynamic_earth_data.h5), and copy it to the folder `data`.

6. Download the pretrained models from this [link](https://syncandshare.lrz.de/getlink/fiWefy8md6CtGLQMG2Mfjc/models.zip), and unzip the content to the folder `models`.

7. Check that everything is ready by running `python check.py`. If the required packages are installed and the data is in place, you will get an `OK`.

8. Start jupyter with the command `jupyter-lab`.


## Data - DynamicEarthNet Challenge (EarthVision - CVPR 2021)

Remote sensing is entering a new era of time-series analysis. Short revisit times of satellites allow for monitoring of many areas across the globe on a weekly basis. However, there has been little exploration of deep learning techniques to leverage this new temporal dimension at scale. Especially, existing approaches have struggled to combine the power of different sensors to make use of all available information. In addition, large scale high quality change detection benchmarks are rare. To stimulate innovation in spatio-temporal machine learning, we have partnered up to propose a unique challenge centered around modeling multi-temporal land cover changes from Planetscope and Sentinel time series data.

Download the data from this [link](https://syncandshare.lrz.de/getlink/fiFXcnxNYWaxALLmRzAq6W/dynamic_earth_data.h5), and copy it to the folder `excelsior-hackatheon/data`.


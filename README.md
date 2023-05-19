![Product Attributes](https://github.com/karanrampal/attribute-prediction/actions/workflows/main.yml/badge.svg)

# Product Attribute Prediction
Detect attributes of the dataset. This dataset consists of images by designers and our goal is to find attributes of these images.

## Usage
First clone the project as follows,
```
git clone <url> <newprojname>
cd <newprojname>
```
Then build the project by using the following command, (assuming build is already installed in your virtual environment, if not then activate your virtual environment and use `conda install build`)
```
make build
```
Next, install the build wheel file as follows,
```
pip install <path to wheel file>
```
Then create the dataset by running the following command (this needs to be done only once, and can be done at anytime after cloning this repo),
```
./src/create_castors.py -r <path to root dir of images> -p <project name> -o <output directory path>
./src/create_dataset.py -c <path to castors> -p <path to pim table> -d <path to padma table> -o <output dir path>
```
Then to start training on a single node with multiple gpu's we can do the following,
```
python <path to train.py> --args1 --args2
```

## Requirements
I used Anaconda with python3, but used pip to install the libraries so that they worked with my multi GPU compute environment in GCP

```
make install
conda activate attrpred-env
```

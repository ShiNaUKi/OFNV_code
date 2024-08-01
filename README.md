# [OFVN:Online Learning in Streaming Features with Nominal Variables]

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## abstract
To address this challenge, we propose a novel Online Learning in Streaming Features with Nominal Variables (OFNV) algorithm, which maps observed mixed and nominal features with arbitrary marginals onto a latent continuous space.
OFNV is based on two primary ideas: 1) employing multi-position probability encoding to preserve the intrinsic structural information of nominal features; 2) using a mixed Gaussian Copula to harmonize the measures of nominal and numerical features, while learning feature correlation information in real-time and utilizing an ensemble method for online model stability updates.
Specially, our approach makes no assumptions on the data marginals nor does it require tuning any hyperparameters.
To evaluate our OFNV, we benchmarked it against state-of-the-art algorithms on ten real-world datasets, and both theoretical analysis and extensive experiments have validated the effectiveness of our proposed method.
## Requiremnt

The code was tested on:

- scipy
- statsmodels
- numpy
- sklearn
- pandas
- math
- matplotlib
- gcimpute

To install requirement:
```
# install requirement
pip install -r requirements
```

## Run
Move to source code directory:
```
cd source
```
Run main.py 
```
python cap_main.py 
```

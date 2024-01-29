# Transfer Learning for EEG

Welcome! This repo aims to achieve simple contemporary deep transfer learning for EEG analysis, specifically brain-computer interface (BCI) applications.
The official implementation of our paper [`T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs`](https://ieeexplore.ieee.org/abstract/document/10210666) (**IEEE TBME, 2023**)

**News**: The implementation for CR and DPL (our papers currently under review) will be updated once when the papers are accepted. They are all implemented under this identical framework for easier reproduction.

## Steps for Usage:

#### 1. Install Dependencies

Install Conda dependencies based on  `environment.yml` file.

#### 2. Download Datasets

To download datasets, run   
```sh 
sh prepare_data.sh
```   

#### (Optional) 3. Training Source Subject Models

We have provided the source models (baseline source-combined EA+EEGNet) under ./runs, but feel free to train them from scratch.  
To train your own source models, run   
```sh 
sh train.sh
```   
or   
```sh 
python ./tl/dnn.py
```  

Note that such source models serve as EEGNet baselines, and are also used in SFUDA and TTA approaches as the initializations. So to save time for TTA/SFUDA for target subject adaptation, it is better to have them ready first.  

Note also that we did not provide non-EA models, and please change code accordingly for TTA approaches under train_target() function when loading pretrained weights.

#### 4. Transfer Learning for Target Subject

To test the T-TIME algorithm, run   
```sh 
sh test.sh
```   
or   
```sh 
python ./tl/ttime.py
```   

Other approaches can be executed in a similar way. Run any of   
```sh 
python ./tl/*.py
```   
for its corresponding results.

Note that ensemble is seperated. For ensemble results, after running T-TIME, run  
```sh 
python ./tl/ttime_ensemble.py
```   

For the machine learning approaches without neural network models, e.g., CSP. Run   
```sh 
python ./ml/feature.py
```

## Hyperparameters

Most hyperparameters/configurations of approaches/experiments are under the *args* variable in the "main" function of each file, and naming should be self-explanatory.


## Currently Implemented Approaches:

#### *. T-TIME
#### 0. EA
#### 1. EEGNet
#### 2. DAN
#### 3. JAN 
#### 4. DANN
#### 5. CDAN
#### 6. MDD
#### 7. MCC
#### 8. SHOT
#### 9. BN-adapt
#### 10. Tent
#### 11. PL
#### 12. T3A
#### 13. CoTTA
#### 14. SAR
#### More to come!

## Contact

Please contact me at syoungli@hust.edu.cn or lsyyoungll@gmail.com for any questions regarding the paper, and use Issues for any questions regarding the code.

## Citation

If you find this repo helpful, please cite our work:
```
@Article{Li2024,
  author  = {Li, Siyang and Wang, Ziwei and Luo, Hanbin and Ding, Lieyun and Wu, Dongrui},
  journal = {IEEE Transactions on Biomedical Engineering},
  title   = {{T}-{TIME}: Test-Time Information Maximization Ensemble for Plug-and-Play {BCI}s},
  year    = {2024},
  number  = {2},
  pages   = {423-432},
  volume  = {71},
  doi     = {10.1109/TBME.2023.3303289},
}
```

## Acknowledgements

All credit of the base framework goes to [`Wen Zhang`](https://github.com/chamwen), do check out the [`Negative Transfer`](https://github.com/chamwen/NT-Benchmark) project.
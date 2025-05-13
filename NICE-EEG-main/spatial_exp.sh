#!/bin/sh

python nice_stand.py --dataset_path ../Preprocessed_data_250Hz --config GA
echo "^ GA ^"
python nice_stand.py --dataset_path ../Preprocessed_data_250Hz --config SA
echo "^ SA ^"
python nice_stand.py --dataset_path ../Preprocessed_data_250Hz --config SAGA
echo "^ SAGA ^"
python nice_stand.py --dataset_path ../Preprocessed_data_250Hz --config GASA
echo "^ GASA ^"
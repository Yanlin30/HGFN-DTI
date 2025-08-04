# from Proposed_HigherOrderMixhop import train_model
import os
# os.environ['CUDA_VISIBLE_DEVICES']="-1"
from Proposed_HigherOrderMixhop_WithNodeAsGraph import train_model
from seed_setting_for_reproducibility import seed_torch
import torch

import random

if __name__ == '__main__':
    numperOfExperiments = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #   # Hyperparameters batch_size #For BindingDB_Kd=512, DAVIS-128/32 , KIBA=32
    for i in range(numperOfExperiments):
        random_seed = i + 1
        print(f"random seed: {random_seed}")
        seed_torch(random_seed)
        train_model('DAVIS', device, binarizeThreshold=7.5, subsampling=1.0,
                    batch_size=128, num_epochs=200)
        seed_torch(random_seed)
        train_model('KIBA', device, binarizeThreshold=12.1, subsampling=1.0,
                    use_pretrained=False, batch_size=128, num_epochs=200,
                    pretrained_dataset_name='KIBA')
        seed_torch(random_seed)
        train_model('BindingDB_Kd', device, binarizeThreshold=8.5,
                    subsampling=1.0, batch_size=128, num_epochs=200,
                    pretrained_dataset_name='DAVIS')
        print(f"Iteration {i + 1} completed")

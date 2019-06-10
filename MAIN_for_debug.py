
from LOW_LEVEL_UTILITIES.utilities import show_batch, save_obj, load_obj, dataset_in_memory, check_datasets
from LOW_LEVEL_UTILITIES.utilities import train_one_epoch, evaluate_one_epoch, test_model
from simulation_dictionary import SimulationDictionary 
from VAE.vae_model import Compositional_VAE

import numpy as np
import os
import torch
import pyro
from pyro.infer import SVI, TraceGraph_ELBO, TraceEnum_ELBO, config_enumerate, JitTraceEnum_ELBO 
from pyro.optim import Adam, Adamax, SGD



#### Create the simulation parameters dictionary
hyper_params=SimulationDictionary()
hyper_params['UNET.N_prediction_maps']=1
hyper_params['UNET.N_up_conv']=2
print(hyper_params)
hyper_params.check_consistency()


# ## Instantiate and Checks the Compositional_VAE
vae = Compositional_VAE(hyper_params)

# ## Load the data
disk_data_dir = "/home/ldalessi/DATA/MULTI_DISK/"
real_data_dir = "/home/ldalessi/DATA/DAPI_ONLY_v3/"
mMNIST_data_dir = "/home/ldalessi/DATA/MULTI_MNIST/"

train_dataset = dataset_in_memory(disk_data_dir,"multi_disk_train_v1",use_cuda=hyper_params['use_cuda'])
test_dataset  = dataset_in_memory(disk_data_dir,"multi_disk_test_v1",use_cuda=hyper_params['use_cuda'])
#train_dataset = dataset_in_memory(mMNIST_data_dir,"multi_mnist_train_large",use_cuda=hyper_params['use_cuda'])
#test_dataset  = dataset_in_memory(mMNIST_data_dir,"multi_mnist_test_large",use_cuda=hyper_params['use_cuda'])

# Set up pyro environment
pyro.clear_param_store()
pyro.set_rng_seed(0)

TEST_FREQUENCY = 5
WRITE_FREQUENCY = 20
smoke_test= False
if(smoke_test):
    pyro.enable_validation(True)
    pyro.distributions.enable_validation(True)
    NUM_EPOCHS = 21
else:
    pyro.enable_validation(False)
    pyro.distributions.enable_validation(False)
    NUM_EPOCHS = 101


# ## Initialize stuff (always) from the same state
min_loss = 99999999
history_dict = {
    "train_loss" : [],
    "test_loss" : [],
    "fg_mu" : [],
    "bg_mu" : [],
    "fg_sigma" : [],
    "bg_sigma" : [],
    "normal_sigma" : [],
    "std_bx_dimfull" : [],
    "std_by_dimfull" : [],
    "std_bw_dimfull" : [],
    "std_bh_dimfull" : []
    }

# ## Set up the names
write_dir  = '/home/ldalessi/REPOS/spacetx-research/ARCHIVE/'
descriptor        = "DISK_v2"
name_vae          = descriptor+"_vae"
name_history      = descriptor+"_hystory"
name_hyper_params = descriptor+"_hyper_params"


# batch size
batch_size = 128

# setup the optimizer
#optimizer = Adamax(adam_args)
adam_args = {"lr": 1.0e-3} # pyro.tutorail has 1E-4 with Adam. I have used Adamax and 1E-3 for a while
optimizer = Adam(adam_args)

svi = SVI(vae.model, config_enumerate(vae.guide, "parallel"), optimizer, loss=TraceEnum_ELBO(max_plate_nesting=2))
#svi = SVI(vae.model, config_enumerate(vae.guide, "parallel"), optimizer, loss=JitTraceEnum_ELBO(max_plate_nesting=2))


# ## Hyperparameter for training (change if necessary)
vae.p_corr_factor = 0.5

vae.lambda_small_box_size  = 0.0
vae.lambda_big_mask_volume = 1.0
vae.lambda_tot_var         = 0.0
vae.lambda_overlap         = 0.0

vae.LOSS_ZMASK = 0.1
vae.LOSS_ZWHAT = 1.0


# # Actual train loop

#with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
# training loop
for epoch in range(0,1):
    vae.train()            
    loss = train_one_epoch(svi, train_dataset, epoch, batch_size, verbose=(epoch==0))
    print("[epoch %03d] train loss: %.4f" % (epoch, loss))
            
    history_dict["train_loss"].append(loss)
    try:
        history_dict["fg_mu"].append(pyro.param("fg_mu").item())
        history_dict["bg_mu"].append(pyro.param("bg_mu").item())
        history_dict["fg_sigma"].append(pyro.param("fg_sigma").item())
        history_dict["bg_sigma"].append(pyro.param("bg_sigma").item())
        history_dict["normal_sigma"].append(pyro.param("normal_sigma").item())
        history_dict["std_bx_dimfull"].append(pyro.param("std_bx_dimfull").item())
        history_dict["std_by_dimfull"].append(pyro.param("std_by_dimfull").item())
        history_dict["std_bw_dimfull"].append(pyro.param("std_bw_dimfull").item())
        history_dict["std_bh_dimfull"].append(pyro.param("std_bh_dimfull").item())
    except:
        pass
        
    if(epoch % TEST_FREQUENCY == 0):
        vae.eval()
        loss = evaluate_one_epoch(svi, test_dataset, epoch, batch_size, verbose=(epoch==0))        
        history_dict["test_loss"].append(loss)
        
        if(loss < min_loss):
            min_loss = loss
            print("[epoch %03d] test  loss: %.4f --New Record--" % (epoch, loss)) 
        else:
            print("[epoch %03d] test  loss: %.4f " % (epoch, loss))
            
        if((loss == min_loss) or ((epoch % WRITE_FREQUENCY) == 0)):   
            # Save on disk
            vae.save_everything(write_dir,name_vae+"_"+str(epoch))
            save_obj(history_dict,write_dir,name_history+"_"+str(epoch))
            save_obj(hyper_params,write_dir,name_hyper_params+"_"+str(epoch))
#print(prof)

#!/usr/bin/env python
# coding: utf-8

import neptune
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from MODULES.utilities_neptune import *
from MODULES.utilities import *
from MODULES.vae_model import *
from MODULES.namedtuple import PreProcess, ImageBbox

# Check versions
from platform import python_version
print("python_version() ---> ",python_version())
print("torch.__version__ --> ",torch.__version__)


params = load_json_as_dict("./ML_parameters.json")

neptune.set_project(params["neptune_project"])
exp = neptune.create_experiment(params=flatten_dict(params),
                                upload_source_files=["./MODULES/vae_model.py", "./encoders_decoders.py"],
                                upload_stdout=True,
                                upload_stderr=True)


# Get the training and test data
preprocessed = load_obj("./data_train.pt")
img_torch = preprocessed.img.float()
roi_mask_torch = preprocessed.roi_mask.bool()
assert len(img_torch.shape) == len(roi_mask_torch.shape) == 4
print("GPU GB after opening data ->",torch.cuda.memory_allocated()/1E9)


BATCH_SIZE = params["simulation"]["batch_size"]
SIZE_CROPS = params["input_image"]["size_raw_image"]
N_TEST = params["simulation"]["N_test"]
N_TRAIN = params["input_image"]["N_train"]
conditional_crop_test = ConditionalRandomCrop(desired_w=SIZE_CROPS, desired_h=SIZE_CROPS, 
                                              min_roi_fraction=0.9, n_crops_per_image=N_TEST)

conditional_crop_train = ConditionalRandomCrop(desired_w=SIZE_CROPS, desired_h=SIZE_CROPS, 
                                               min_roi_fraction=0.9, n_crops_per_image=N_TRAIN)

test_data = conditional_crop_test.forward(img=img_torch,
                                          roi_mask=roi_mask_torch)
print("GPU GB after defining test data ->",torch.cuda.memory_allocated()/1E9)


test_loader = SpecialDataSet(img=test_data,
                             store_in_cuda=False,
                             shuffle=False,
                             drop_last=False,
                             batch_size=BATCH_SIZE)
test_batch_example = test_loader.check_batch()
exp.log_image("test_batch_example", test_batch_example)
print("GPU GB after test_loader ->",torch.cuda.memory_allocated()/1E9)


train_loader = SpecialDataSet(img=img_torch, 
                              roi_mask=roi_mask_torch,
                              data_augmentation=conditional_crop_train,
                              store_in_cuda=False,
                              shuffle=True,
                              drop_last=True,
                              batch_size=BATCH_SIZE)
traing_batch_example = traing_loader.check_batch()
exp.log_image("traing_batch_example", traing_batch_example)
print("GPU GB after train_loader ->",torch.cuda.memory_allocated()/1E9)

reference_imgs, labels, index = test_loader.load(8)
tmp = show_batch(reference_imgs, n_padding=4, figsize=(12,12), title='reference imgs')
exp.log_image("reference_imgs", tmp)


# Instantiate model, optimizer and checks
vae = CompositionalVae(params)
optimizer = instantiate_optimizer(model=vae, dict_params_optimizer=params["optimizer"])
print("GPU GB after model and optimizer ->",torch.cuda.memory_allocated()/1E9)

imgs_out = vae.inference_and_generator.unet.show_grid(reference_imgs)
unet_grid = show_batch(imgs_out[:,0])
exp.log_image("unet_grid", unet_grid)

vae.eval()
if torch.cuda.is_available():
    imgs_in = imgs_in.cuda()
generated_data = vae.generate(imgs_in=reference_imgs, draw_boxes=True, draw_bg=False)
tmp_img = show_batch(generated_data.imgs_rec[:8], title="untrained generator", figsize=(12,12))
tmp_prob = show_batch(generated_data.inference.p_map[:8], n_padding=2, title="untrained probability", figsize=(12,12))
exp.log_image("untrained_generator_img", tmp_img)
exp.log_image("untrained_generator_prob", tmp_prob)
print("GPU GB after generator ->",torch.cuda.memory_allocated()/1E9)

# Check the constraint dictionary
print("simulation type = "+str(params["simulation"]["type"]))
    
if (params["simulation"]["type"] == "scratch"):
    
    epoch_restart = -1
    history_dict = {}
    min_test_loss = 99999999

elif (params["simulation"]["type"] == "resume"):
    
    resumed = file2resumed(path=ckpt_file, device=None)
    #resumed = file2resumed(path=ckpt_file, device='cpu')
        
    load_model_optimizer(resumed=resumed,  
                         model=vae,
                         optimizer=optimizer,
                         overwrite_member_var=True)
    
    ckp = load_info(resumed=resumed, 
                    load_epoch=True, 
                    load_history=True)
    
    epoch_restart = ckp.epoch
    history_dict = ckp.history_dict
    min_test_loss = min(history_dict["test_loss"])
    
elif (params["simulation"]["type"] == "pretrained"):
    
    resumed = file2resumed(path=ckpt_file, device=None)
    # resumed = file2resumed(path=ckpt_file, device='cpu')
        
    load_model_optimizer(resumed=resumed,  
                         model=vae,
                         optimizer=None,
                         overwrite_member_var=False)
       
    epoch_restart = -1
    history_dict = {}
    min_test_loss = 99999999
    
else:
    raise Exception("simulation type is NOT recognized")
    
# instantiate the scheduler if necessary    
if params["optimizer"]["scheduler_is_active"]:
    scheduler = instantiate_scheduler(optimizer=optimizer, dict_params_scheduler=params["optimizer"])


TEST_FREQUENCY = params["simulation"]["TEST_FREQUENCY"]
CHECKPOINT_FREQUENCY = params["simulation"]["CHECKPOINT_FREQUENCY"]
NUM_EPOCHS = params["simulation"]["MAX_EPOCHS"]
torch.cuda.empty_cache()
for delta_epoch in range(1,NUM_EPOCHS+1):
    epoch = delta_epoch+epoch_restart    
    
    vae.prob_corr_factor=linear_interpolation(epoch, 
                                              values=params["shortcut_prob_corr_factor"]["values"],
                                              times=params["shortcut_prob_corr_factor"]["times"])
    exp.log_metric("prob_corr_factor", vae.prob_corr_factor)
        
    with torch.autograd.set_detect_anomaly(False):
        with torch.enable_grad():
            vae.train()
            train_metrics = process_one_epoch(model=vae, 
                                              dataloader=train_loader, 
                                              optimizer=optimizer, 
                                              verbose=(epoch==0), 
                                              weight_clipper=None)

            if params["optimizer"]["scheduler_is_active"]:
                scheduler.step()
            
            with torch.no_grad():
                print("Train "+train_metrics.pretty_print(epoch))
                log_metrics(exp, train_metrics, prefix="train_") 
                history_dict = append_dict_to_dict(source=train_metrics, 
                                                   target=history_dict,
                                                   prefix_exclude="wrong_examples",
                                                   prefix_to_add="train_")
        
    


                if(epoch % TEST_FREQUENCY == 0):
                    output = vae.forward(reference_imgs, draw_image=True, draw_boxes=True, verbose=False) 
                    imgs_rec_train = show_batch(output.imgs[:8],n_col=4,n_padding=4,title="TRAIN MODE, EPOCH = "+str(epoch))
                    p_map_train = show_batch(output.inference.p_map[:8],n_col=4,n_padding=4,title="TRAIN MODE, EPOCH = "+str(epoch), normalize_range=None)
                    bg_train = show_batch(output.inference.big_bg[:8],n_col=4,n_padding=4,title="TRAIN MODE, EPOCH = "+str(epoch))
                    exp.log_image("rec_imgs_train", imgs_rec_train)
                    exp.log_image("rec_prob_train", p_map_train)
                    exp.log_image("rec_bg_train", bg_train)

                    vae.eval()
                    test_metrics = process_one_epoch(model=vae, 
                                                     dataloader=test_loader, 
                                                     optimizer=optimizer, 
                                                     verbose=(epoch==0), 
                                                     weight_clipper=None)
                    print("Test  "+test_metrics.pretty_print(epoch))
                    log_metrics(exp, test_metrics, prefix="test_") 
                    history_dict = append_dict_to_dict(source=test_metrics, 
                                                       target=history_dict,
                                                       prefix_exclude="wrong_examples",
                                                       prefix_to_add="test_")
        
                    output = vae.forward(reference_imgs, draw_imgs=True, draw_boxes=True, verbose=False)
                    imgs_rec_test = show_batch(output.imgs[:8],n_col=4,n_padding=4,title="TEST MODE, EPOCH = "+str(epoch))
                    p_map_test = show_batch(output.inference.p_map[:8],n_col=4,n_padding=4,title="TEST MODE, EPOCH = "+str(epoch), normalize_range=None)
                    bg_test = show_batch(output.inference.big_bg[:8],n_col=4,n_padding=4,title="TEST MODE, EPOCH = "+str(epoch))
                    exp.log_image("rec_imgs_test", imgs_rec_test)
                    exp.log_image("rec_prob_test", p_map_test)
                    exp.log_image("rec_bg_test", bg_test)

                    segmentation = vae.segment(imgs_in=reference_imgs)
                    seg_int_mask = show_batch(segmentation.integer_mask, n_padding=4, figsize=(12,12), title='epoch= {0:6d}'.format(epoch))
                    seg_fg_prob = show_batch(segmentation.fg_prob, n_padding=4, figsize=(12,12), title='epoch= {0:6d}'.format(epoch))
                    exp.log_image("seg_integer_mask", seg_int_mask)
                    exp.log_image("seg_fg_prob", seg_fg_prob)
                    
                    generated = vae.generate(imgs_in=reference_imgs, draw_boxes=True)
                    gen_img = show_batch(generated.imgs[:8], n_padding=4, figsize=(12,12), title='epoch= {0:6d}'.format(epoch))
                    gen_prob = show_batch(generated.inference.p_map[:8], n_padding=4, figsize=(12,12), title='epoch= {0:6d}'.format(epoch))
                    gen_big_masks = show_batch(generated.inference.big_mask[0, :8], n_padding=4, figsize=(12,12), title='epoch= {0:6d}'.format(epoch))
                    gen_big_imgs = show_batch(generated.inference.big_imgs[0, :8], n_padding=4, figsize=(12,12), title='epoch= {0:6d}'.format(epoch))
                    exp.log_image("generated_img", gen_img)
                    exp.log_image("generated_prob", gen_prob)
                    exp.log_image("generated_big_masks", gen_big_masks)
                    exp.log_image("generated_big_imgs", gen_big_imgs)

                    if((test_loss == min_test_loss) or ((epoch % CHECKPOINT_FREQUENCY) == 0)): 
                        ckpt = create_ckpt(model=vae, 
                                   optimizer=optimizer, 
                                   epoch=epoch, 
                                   hyperparams_dict=params)
                        save_obj(ckpt, "last_ckpt.pt")  # save locally to file 
                        log_last_ckpt(exp, "last_ckpt.pt")  # log file into neptune


# # Check segmentation WITH tiling
img_to_segment = train_loader.img[0,:,1000:1300,2100:2400]
roi_mask_to_segment = train_loader.roi_mask[0,:,1000:1300,2100:2400]
tiling = vae.segment_with_tiling(single_img=img_to_segment,
                                 roi_mask=roi_mask_to_segment,
                                 crop_size=None,
                                 stride=(40,40),
                                 n_objects_max_per_patch=None,
                                 prob_corr_factor=None,
                                 overlap_threshold=None,
                                 radius_nn=10,
                                 batch_size=64)
save_obj(tiling, "tiling.pt")  # save locally to file 
exp.log_artifact("tiling", "tiling.pt")  # log gile into neptune

figure, axes = plt.subplots(ncols=4, figsize=(24,24))
axes[0].imshow(skimage.color.label2rgb(tiling.integer_mask[0,0].cpu().numpy(),
                                       numpy.zeros_like(tiling.integer_mask[0,0].cpu().numpy()),
                                       alpha=1.0,
                                       bg_label=0))
axes[1].imshow(skimage.color.label2rgb(tiling.integer_mask[0,0].cpu().numpy(),
                                         tiling.raw_image[0,0].cpu().numpy(),
                                         alpha=0.25,
                                         bg_label=0))
axes[2].imshow(tiling.fg_prob[0,0].cpu().numpy(), cmap='gray')
axes[3].imshow(tiling.raw_image[0].cpu().permute(1,2,0).squeeze(-1).numpy(), cmap='gray')

axes[0].set_title("sample integer mask")
axes[1].set_title("sample integer mask")
axes[2].set_title("fg prob")
axes[3].set_title("raw image")
exp.log_image("tiling_img", figure)


#####plt.imshow(output_test.inference.p_map[chosen,0].cpu().numpy())
#####_ = plt.colorbar()
#####print(torch.topk(output_test.inference.p_map[chosen,0].view(-1), k=10, largest=True, sorted=True)[0])
#####
#####
###### In[ ]:
#####
#####
#####_ = plt.hist(output_train.inference.p_map[0,0].view(-1).cpu().numpy(), density=True, bins=50, label="pmap_train")
#####_ = plt.hist(output_test.inference.p_map[0,0].view(-1).cpu().numpy(), density=True, bins=50, label="pmap_test")
#####plt.legend()
#####plt.savefig(os.path.join(dir_output, "hist_pmap.png"))
#####
#####
###### # Visualize one chosen image in details
#####
###### In[ ]:
#####
#####
#####output = output_train
#####how_many_to_show=20
#####counts = torch.sum(output.inference.prob>0.5,dim=0).view(-1).cpu().numpy().tolist()
#####prob_tmp = np.round(output.inference.prob[:how_many_to_show,chosen].view(-1).cpu().numpy(),decimals=4)*10000
#####prob_title = (prob_tmp.astype(int)/10000).tolist()
#####print("counts ->",counts[chosen]," prob ->",prob_title)
#####
#####
###### In[ ]:
#####
#####
#####tmp1 = reference_imgs[chosen]
#####ch_out = tmp1.shape[-3]
#####tmp2 = torch.sum(output.inference.big_img[:how_many_to_show,chosen],dim=0).expand(ch_out,-1,-1)
#####tmp3 = torch.sum(output.inference.big_mask[:how_many_to_show,chosen],dim=0).expand(ch_out,-1,-1)
#####mask_times_imgs = output.inference.big_mask * output.inference.big_img
#####tmp4 = torch.sum(mask_times_imgs[:how_many_to_show,chosen],dim=0).expand(ch_out,-1,-1)
#####print("sum big_masks", torch.max(tmp3))
#####print("sum big_masks * big_imgs", torch.max(tmp4))
#####combined = torch.stack((tmp1,tmp2,tmp3,tmp4),dim=0)
#####print(combined.shape)
#####b = show_batch(combined, n_col=2, title="# ref, IMGS, MASKS, IMGS*MASKS", figsize=(24,24))
#####b.savefig(os.path.join(dir_output, "ref_img_mask.png"))
#####display(b)
#####
#####
###### In[ ]:
#####
#####
#####print(torch.min(output.inference.big_mask[:how_many_to_show,chosen]), torch.max(output.inference.big_mask[:how_many_to_show,chosen]))
#####show_batch(output.inference.big_mask[:how_many_to_show,chosen], n_col=4, title="# MASKS", figsize=(24,24))
#####
#####
###### In[ ]:
#####
#####
#####b = show_batch(reference_imgs[chosen]+output.inference.big_mask[:how_many_to_show,chosen], 
#####               n_col=3, n_padding=4,title="# MASKS over REF, p="+str(prob_title), figsize=(24,24))
#####b.savefig(os.path.join(dir_output, "mask_over_ref.png"))
#####display(b)
#####
#####
###### In[ ]:
#####
#####
#####b = show_batch(reference_imgs[chosen]+10*output.inference.big_img[:how_many_to_show,chosen], 
#####               n_col=4, n_padding=4,title="# IMGS over REF, p="+str(prob_title), figsize=(24,24), normalize_range=(0,1))
#####b.savefig(os.path.join(dir_output, "imgs_over_ref.png"))
#####display(b)
#####
#####
###### In[ ]:
#####
#####
#####output.inference.prob.shape
#####
#####
###### In[ ]:
#####
#####
#####prob =  output.inference.prob[:,chosen, None, None, None]
#####
#####b_img = output.inference.big_img[:,chosen]
#####ch_out = b_img.shape[-3]
#####b_mask = output.inference.big_mask[:,chosen].expand(-1,ch_out,-1,-1)
#####b_combined = b_img * b_mask * prob
#####tmp = torch.cat((b_mask, b_img, b_combined), dim=0)
#####b = show_batch(tmp, n_col=tmp.shape[0]//3, n_padding=4, title="# mask, imgs, product. p="+str(prob_title), figsize=(24,24))
#####b.savefig(os.path.join(dir_output, "mask_imgs_product.png"))
#####display(b)
#####
#####
###### ### Show the probability map
#####
###### In[ ]:
#####
#####
#####_ = plt.imshow(output.inference.p_map[chosen,0].cpu().numpy())
#####_ = plt.colorbar()
#####plt.savefig(os.path.join(dir_output, "pmap_chosen.png"))

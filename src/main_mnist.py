#!/usr/bin/env python
# coding: utf-8

import neptune
from MODULES.utilities_neptune import log_object_as_artifact, log_model_summary, log_img_only
from MODULES.utilities_neptune import log_dict_metrics, log_concordance
from MODULES.vae_model import *
from MODULES.utilities_visualization import show_batch, plot_tiling, plot_all_from_dictionary, plot_label_contours
from MODULES.utilities_visualization import plot_reconstruction_and_inference, plot_generation, plot_segmentation
from MODULES.utilities_visualization import plot_img_and_seg, plot_concordance
from MODULES.utilities_ml import ConditionalRandomCrop, SpecialDataSet, process_one_epoch
from MODULES.graph_clustering import GraphSegmentation
from MODULES.utilities import QC_on_integer_mask, concordance_integer_masks, load_json_as_dict
import skimage.io

# Check versions
import torch
import numpy
from platform import python_version
print("python_version() ---> ", python_version())
print("torch.__version__ --> ", torch.__version__)

# make sure to fix the randomness at the very beginning
torch.manual_seed(0)
numpy.random.seed(0)

params = load_json_as_dict("./ML_parameters.json")

neptune.set_project(params["neptune_project"])
exp: neptune.experiments.Experiment = \
    neptune.create_experiment(params=flatten_dict(params),
                              upload_source_files=["./main_mnist.py", "./ML_parameters.json", "./MODULES/vae_parts.py",
                                                   "./MODULES/vae_model.py", "./MODULES/encoders_decoders.py"],
                              upload_stdout=True,
                              upload_stderr=True)

# Get the training and test data
img_train, seg_mask_train, count_train = load_obj("./data_train.pt")
img_test, seg_mask_test, count_test = load_obj("./data_test.pt")
BATCH_SIZE = params["simulation"]["batch_size"]


train_loader = SpecialDataSet(img=img_train,
                              roi_mask=None,
                              seg_mask=seg_mask_train,
                              labels=count_train,
                              batch_size=BATCH_SIZE,
                              drop_last=False,
                              shuffle=True)

train_batch_example_fig = train_loader.check_batch()
log_img_only(name="train_batch_example", fig=train_batch_example_fig, experiment=exp)
if torch.cuda.is_available():
    print("GPU GB after train_loader ->", torch.cuda.memory_allocated()/1E9)

test_loader = SpecialDataSet(img=img_test,
                             roi_mask=None,
                             seg_mask=seg_mask_test,
                             labels=count_test,
                             batch_size=BATCH_SIZE,
                             drop_last=False,
                             shuffle=False)

test_batch_example_fig = test_loader.check_batch()
log_img_only(name="test_batch_example", fig=test_batch_example_fig, experiment=exp)
if torch.cuda.is_available():
    print("GPU GB after train_loader ->", torch.cuda.memory_allocated()/1E9)

# Instantiate model, optimizer and checks
vae = CompositionalVae(params)
log_model_summary(vae)
optimizer = instantiate_optimizer(model=vae, dict_params_optimizer=params["optimizer"])
if torch.cuda.is_available():
    print("GPU GB after vae ->", torch.cuda.memory_allocated()/1E9)

# Make reference images
tmp_imgs, tmp_seg, tmp_count = test_loader.load(index=torch.arange(64))[:3]
mask_5_or_6 = (tmp_count == 5) + (tmp_count == 6)
reference_imgs = tmp_imgs[mask_5_or_6][:16]
reference_seg = tmp_seg[mask_5_or_6][:16]
reference_count = tmp_count[mask_5_or_6][:16]
show_batch(reference_imgs, normalize_range=(0.0, 1.0), neptune_name="reference_imgs")
plot_img_and_seg(img=reference_imgs,
                 seg=reference_seg,
                 figsize=(6, 6*reference_imgs.shape[0]),
                 neptune_name="reference_imgs_and_seg")

if torch.cuda.is_available():
    reference_imgs = reference_imgs.cuda()
imgs_out = vae.inference_and_generator.unet.show_grid(reference_imgs)
unet_grid_fig = show_batch(imgs_out[:, 0], normalize_range=(0.0, 1.0), neptune_name="unet_grid")

# Check the constraint dictionary
print("simulation type = "+str(params["simulation"]["type"]))
    
if params["simulation"]["type"] == "scratch":
    
    epoch_restart = -1
    history_dict = {}
    min_test_loss = 999999

elif params["simulation"]["type"] == "resume":
    
    ckpt = file2ckpt(path="ckpt.pt", device=None)
    # ckpt = file2ckpt(path="ckpt.pt", device='cpu')

    load_from_ckpt(ckpt=ckpt,
                   model=vae,
                   optimizer=optimizer,
                   overwrite_member_var=True)

    epoch_restart = ckpt.get('epoch', -1)
    history_dict = ckpt.get('history_dict', {})
    try:
        min_test_loss = min(history_dict.get("test_loss", 999999))
    except:
        min_test_loss = 999999

elif params["simulation"]["type"] == "pretrained":

    ckpt = file2ckpt(path="ckpt.pt", device=None)
    # ckpt = file2ckpt(path="ckpt.pt", device='cpu')

    load_from_ckpt(ckpt=ckpt,
                   model=vae,
                   optimizer=None,
                   overwrite_member_var=False)
       
    epoch_restart = -1
    history_dict = {}
    min_test_loss = 999999
    
else:
    raise Exception("simulation type is NOT recognized")
    
# instantiate the scheduler if necessary    
if params["optimizer"]["scheduler_is_active"]:
    scheduler = instantiate_scheduler(optimizer=optimizer, dict_params_scheduler=params["optimizer"])


TEST_FREQUENCY = params["simulation"]["TEST_FREQUENCY"]
CHECKPOINT_FREQUENCY = params["simulation"]["CHECKPOINT_FREQUENCY"]
NUM_EPOCHS = params["simulation"]["MAX_EPOCHS"]
torch.cuda.empty_cache()
for delta_epoch in range(1, NUM_EPOCHS+1):
    epoch = delta_epoch+epoch_restart    
    
    vae.prob_corr_factor = linear_interpolation(epoch,
                                                values=params["shortcut_prob_corr_factor"]["values"],
                                                times=params["shortcut_prob_corr_factor"]["times"])
    exp.log_metric("prob_corr_factor", vae.prob_corr_factor)
        
    with torch.autograd.set_detect_anomaly(False):
        with torch.enable_grad():
            vae.train()
            train_metrics = process_one_epoch(model=vae, 
                                              dataloader=train_loader, 
                                              optimizer=optimizer, 
                                              verbose=(epoch == 0),
                                              weight_clipper=None,
                                              neptune_experiment=exp,
                                              neptune_prefix="train_")
            print("Train " + train_metrics.pretty_print(epoch))

            if params["optimizer"]["scheduler_is_active"]:
                scheduler.step()
            
            with torch.no_grad():

                history_dict = append_to_dict(source=train_metrics,
                                              destination=history_dict,
                                              prefix_exclude="wrong_examples",
                                              prefix_to_add="train_")

                if (epoch % TEST_FREQUENCY) == 0:

                    vae.eval()
                    test_metrics = process_one_epoch(model=vae, 
                                                     dataloader=test_loader, 
                                                     optimizer=optimizer, 
                                                     verbose=(epoch == 0),
                                                     weight_clipper=None,
                                                     neptune_experiment=exp,
                                                     neptune_prefix="test_")
                    print("Test  "+test_metrics.pretty_print(epoch))
                    history_dict = append_to_dict(source=test_metrics,
                                                  destination=history_dict,
                                                  prefix_exclude="wrong_examples",
                                                  prefix_to_add="test_")
                    
                    output: Output = vae.forward(reference_imgs, draw_image=True, draw_boxes=True, verbose=False)
                    plot_reconstruction_and_inference(output, epoch=epoch, prefix="rec_")
                    reference_n_cells = output.inference.sample_c.sum().item()
                    tmp_dict = {"reference_n_cells": reference_n_cells,
                                "delta_n_cells": (reference_n_cells - reference_count).sum()}
                    log_dict_metrics(tmp_dict)
                    history_dict = append_to_dict(source=tmp_dict,
                                                  destination=history_dict)

                    segmentation: Segmentation = vae.segment(batch_imgs=reference_imgs)
                    plot_segmentation(segmentation, epoch=epoch, prefix="seg_")

                    # Here I could add a measure of agreement with the ground truth
                    #a = segmentation.integer_mask[0, 0].long()
                    #b = reference_seg.long()
                    #print("CHECK", a.shape, a.dtype, b.shape, b.dtype)
                    #concordance_vs_gt = concordance_integer_masks(a,b)
                    #plot_concordance(concordance=concordance_vs_gt, neptune_name="concordance_vs_gt_")
                    #log_concordance(concordance=concordance_vs_gt, prefix="concordance_vs_gt_")

                    generated: Output = vae.generate(imgs_in=reference_imgs, draw_boxes=True)
                    plot_generation(generated, epoch=epoch, prefix="gen_")

                    test_loss = test_metrics.loss
                    min_test_loss = min(min_test_loss, test_loss)

                    if (test_loss == min_test_loss) or (epoch % CHECKPOINT_FREQUENCY == 0):
                        ckpt = create_ckpt(model=vae,
                                           optimizer=optimizer,
                                           epoch=epoch,
                                           hyperparams_dict=params,
                                           history_dict=history_dict)
                        log_object_as_artifact(name="last_ckpt", obj=ckpt)  # log file into neptune
                        plot_all_from_dictionary(history_dict,
                                                 params,
                                                 test_frequency=TEST_FREQUENCY,
                                                 train_or_test="test",
                                                 verbose=True)
                        plot_all_from_dictionary(history_dict,
                                                 params,
                                                 test_frequency=TEST_FREQUENCY,
                                                 train_or_test="train",
                                                 verbose=True)

exp.stop()

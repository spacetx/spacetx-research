#!/usr/bin/env python
# coding: utf-8

import neptune
from MODULES.utilities_neptune import log_object_as_artifact, log_model_summary, log_img_only
from MODULES.utilities_neptune import log_dict_metrics, log_concordance
from MODULES.vae_model import *
from MODULES.utilities_visualization import show_batch, plot_tiling, plot_all_from_dictionary, plot_label_contours
from MODULES.utilities_visualization import plot_reconstruction_and_inference, plot_generation, plot_segmentation
from MODULES.utilities_visualization import plot_concordance
from MODULES.utilities_ml import ConditionalRandomCrop, SpecialDataSet, process_one_epoch
from MODULES.graph_clustering import GraphSegmentation
from MODULES.utilities import QC_on_integer_mask, concordance_integer_masks
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
                              upload_source_files=["./main.py", "./ML_parameters.json", "./MODULES/vae_parts.py",
                                                   "./MODULES/vae_model.py", "./MODULES/encoders_decoders.py"],
                              upload_stdout=True,
                              upload_stderr=True)

# Get the training and test data
preprocessed = load_obj("./data_train.pt")
img_torch = preprocessed.img.float()
roi_mask_torch = preprocessed.roi_mask.bool()
assert len(img_torch.shape) == len(roi_mask_torch.shape) == 4
# print("GPU GB after opening data ->",torch.cuda.memory_allocated()/1E9)

BATCH_SIZE = params["simulation"]["batch_size"]
SIZE_CROPS = params["input_image"]["size_raw_image"]
N_TEST = params["simulation"]["N_test"]
N_TRAIN = params["simulation"]["N_train"]
conditional_crop_test = ConditionalRandomCrop(desired_w=SIZE_CROPS, desired_h=SIZE_CROPS, 
                                              min_roi_fraction=0.9, n_crops_per_image=N_TEST)

conditional_crop_train = ConditionalRandomCrop(desired_w=SIZE_CROPS, desired_h=SIZE_CROPS, 
                                               min_roi_fraction=0.9, n_crops_per_image=N_TRAIN)

test_data = conditional_crop_test.crop(img=img_torch,
                                       roi_mask=roi_mask_torch)
# print("GPU GB after defining test data ->",torch.cuda.memory_allocated()/1E9)


test_loader = SpecialDataSet(img=test_data,
                             store_in_cuda=False,
                             shuffle=False,
                             drop_last=False,
                             batch_size=BATCH_SIZE)
test_batch_example_fig = test_loader.check_batch()
log_img_only(name="test_batch_example", fig=test_batch_example_fig, experiment=exp)

train_loader = SpecialDataSet(img=img_torch,
                              roi_mask=roi_mask_torch,
                              data_augmentation=conditional_crop_train,
                              store_in_cuda=False,
                              shuffle=True,
                              drop_last=True,
                              batch_size=BATCH_SIZE)
train_batch_example_fig = train_loader.check_batch()
log_img_only(name="train_batch_example", fig=train_batch_example_fig, experiment=exp)
# print("GPU GB after train_loader ->",torch.cuda.memory_allocated()/1E9)

# Make a batch of reference images by cropping the train_data at consecutive locations
reference_imgs_list = []
crop_size = params["input_image"]["size_raw_image"]
for ni in range(2):
    i = 1080 + ni * crop_size
    for nj in range(4):
        j = 2140 + nj * crop_size
        reference_imgs_list.append(img_torch[..., i:i+crop_size, j:j+crop_size])
reference_imgs = torch.cat(reference_imgs_list, dim=-4)
if torch.cuda.is_available():
    reference_imgs = reference_imgs.cuda()
_ = show_batch(reference_imgs,
               n_padding=4,
               figsize=(12, 12),
               title="reference imgs",
               neptune_name="reference_imgs")

# Instantiate model, optimizer and checks
vae = CompositionalVae(params)
log_model_summary(vae)
optimizer = instantiate_optimizer(model=vae, dict_params_optimizer=params["optimizer"])

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
                    reference_n_cells = output.inference.sample_c_map.sum().item()
                    tmp_dict = {"reference_n_cells": reference_n_cells}
                    log_dict_metrics(tmp_dict)
                    history_dict = append_to_dict(source=tmp_dict,
                                                  destination=history_dict)

                    segmentation: Segmentation = vae.segment(batch_imgs=reference_imgs)
                    plot_segmentation(segmentation, epoch=epoch, prefix="seg_")

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

# # Check segmentation WITH and WITHOUT tiling to the GROUND_TRUTH
img_to_segment = train_loader.img[0, :, 940:1240, 2140:2440]
roi_mask_to_segment = train_loader.roi_mask[0, :, 940:1240, 2140:2440]
gt_numpy = skimage.io.imread("./ground_truth").astype(numpy.int32)[940:1240, 2140:2440]

# tiling segmentation
tiling: Segmentation = vae.segment_with_tiling(single_img=img_to_segment,
                                               roi_mask=roi_mask_to_segment,
                                               crop_size=None,
                                               stride=(40, 40),
                                               n_objects_max_per_patch=None,
                                               prob_corr_factor=None,
                                               overlap_threshold=None,
                                               radius_nn=10,
                                               batch_size=64)
# log_object_as_artifact(name="tiling", obj=tiling, verbose=True)
tiling_fig = plot_tiling(tiling, neptune_name="tiling_before_graph")

# perform graph analysis
g = GraphSegmentation(tiling, min_fg_prob=0.1, min_edge_weight=0.01, normalize_graph_edges=True)
partition_graph = g.find_partition_leiden(resolution=1.0,
                                          window=None,
                                          min_size=30,
                                          cpm_or_modularity="modularity",
                                          each_cc_separately=False,
                                          n_iterations=10,
                                          initial_membership=None)
g.plot_partition(partition_graph, neptune_name="tiling_after_graph")
graph_integer_mask = g.partition_2_integer_mask(partition_graph)
if torch.cuda.is_available():
    graph_integer_mask = graph_integer_mask.cuda()

# qualitative comparison segmentation
simple_integer_mask = QC_on_integer_mask(tiling.integer_mask[0, 0], min_area=30).to(dtype=graph_integer_mask.dtype,
                                                                                    device=graph_integer_mask.device)

gt_integer_mask = torch.from_numpy(QC_on_integer_mask(gt_numpy, min_area=30)).to(dtype=graph_integer_mask.dtype,
                                                                                 device=graph_integer_mask.device)

# compare with ground truth
# print("graph_integer_mask", graph_integer_mask.device, graph_integer_mask.dtype, graph_integer_mask.shape)
# print("gt_integer_mask", gt_integer_mask.device, gt_integer_mask.dtype, gt_integer_mask.shape)
# print("simple_integer_mask", simple_integer_mask.device, simple_integer_mask.dtype, simple_integer_mask.shape)

plot_label_contours(label=gt_integer_mask,
                    image=tiling.raw_image[0, 0],
                    contour_thickness=2,
                    neptune_name="tiling_contours_ground_truth")

plot_label_contours(label=simple_integer_mask,
                    image=tiling.raw_image[0, 0],
                    contour_thickness=2,
                    neptune_name="tiling_contours_simple")

plot_label_contours(label=graph_integer_mask,
                    image=tiling.raw_image[0, 0],
                    contour_thickness=2,
                    neptune_name="tiling_contours_graph")

# quantitative comparison
simple_vs_gt = concordance_integer_masks(simple_integer_mask, gt_integer_mask)
graph_vs_gt = concordance_integer_masks(graph_integer_mask, gt_integer_mask)
simple_vs_graph = concordance_integer_masks(simple_integer_mask, graph_integer_mask)

plot_concordance(concordance=simple_vs_gt, neptune_name="concordance_simple_vs_gt_")
plot_concordance(concordance=graph_vs_gt, neptune_name="concordance_graph_vs_gt_")
plot_concordance(concordance=simple_vs_graph, neptune_name="concordance_simple_vs_graph_")

log_concordance(concordance=simple_vs_gt, prefix="concordance_simple_vs_gt_")
log_concordance(concordance=graph_vs_gt, prefix="concordance_graph_vs_gt_")
log_concordance(concordance=simple_vs_graph, prefix="concordance_simple_vs_graph_")

exp.stop()

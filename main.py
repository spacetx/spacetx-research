#!/usr/bin/env python
# coding: utf-8

import neptune
from MODULES.utilities_neptune import log_matplotlib_as_png, log_object_as_artifact, log_model_summary, log_last_ckpt
from MODULES.vae_model import *
from MODULES.utilities_visualization import show_batch, plot_tiling, plot_all_from_dictionary
from MODULES.utilities_visualization import plot_reconstruction_and_inference, plot_segmentation, plot_geco_parameters
from MODULES.utilities_ml import ConditionalRandomCrop, SpecialDataSet, process_one_epoch

# Check versions
from platform import python_version
print("python_version() ---> ", python_version())
print("torch.__version__ --> ", torch.__version__)


params = load_json_as_dict("./ML_parameters.json")

neptune.set_project(params["neptune_project"])
exp: neptune.experiments.Experiment = \
    neptune.create_experiment(params=flatten_dict(params),
                              upload_source_files=["./MODULES/vae_model.py", "./MODULES/encoders_decoders.py"],
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
log_matplotlib_as_png("test_batch_example", test_batch_example_fig)

train_loader = SpecialDataSet(img=img_torch,
                              roi_mask=roi_mask_torch,
                              data_augmentation=conditional_crop_train,
                              store_in_cuda=False,
                              shuffle=True,
                              drop_last=True,
                              batch_size=BATCH_SIZE)
train_batch_example_fig = train_loader.check_batch()
log_matplotlib_as_png("train_batch_example", train_batch_example_fig)
# print("GPU GB after train_loader ->",torch.cuda.memory_allocated()/1E9)

reference_imgs, labels, index = test_loader.load(8)
reference_imgs_fig = show_batch(reference_imgs,
                                n_padding=4,
                                figsize=(12, 12),
                                title='reference imgs')
log_matplotlib_as_png("reference_imgs", reference_imgs_fig)

# Instantiate model, optimizer and checks
vae = CompositionalVae(params)
log_model_summary(vae)
optimizer = instantiate_optimizer(model=vae, dict_params_optimizer=params["optimizer"])
# print("GPU GB after model and optimizer ->",torch.cuda.memory_allocated()/1E9)

imgs_out = vae.inference_and_generator.unet.show_grid(reference_imgs)
unet_grid_fig = show_batch(imgs_out[:, 0], normalize_range=(0.0, 1.0), neptune_name="unet_grid")

# Check the constraint dictionary
print("simulation type = "+str(params["simulation"]["type"]))
    
if params["simulation"]["type"] == "scratch":
    
    epoch_restart = -1
    history_dict = {}
    min_test_loss = 99999999

elif params["simulation"]["type"] == "resume":
    
    ckpt = file2ckpt(path="ckpt.pt", device=None)
    # ckpt = file2ckpt(path="ckpt.pt", device='cpu')

    load_from_ckpt(ckpt=ckpt,
                   model=vae,
                   optimizer=optimizer,
                   overwrite_member_var=True)

    epoch_restart = ckpt.epoch
    history_dict = ckpt.history_dict
    min_test_loss = min(history_dict["test_loss"])
    
elif params["simulation"]["type"] == "pretrained":

    ckpt = file2ckpt(path="ckpt.pt", device=None)

    load_from_ckpt(ckpt=ckpt,
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
NUM_EPOCHS = 2 #params["simulation"]["MAX_EPOCHS"]
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
                    output: Output = vae.forward(reference_imgs, draw_image=True, draw_boxes=True, verbose=False)
                    plot_reconstruction_and_inference(output, epoch=epoch, prefix="rec_", postfix="_train")

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
                    plot_reconstruction_and_inference(output, epoch=epoch, prefix="rec_", postfix="_test")

                    segmentation: Segmentation = vae.segment(batch_imgs=reference_imgs)
                    plot_segmentation(segmentation, epoch=epoch, prefix="seg_", postfix="_test")

                    generated: Output = vae.generate(imgs_in=reference_imgs, draw_boxes=True)
                    plot_reconstruction_and_inference(generated, epoch=epoch, prefix="gen_", postfix="_test")

                    test_loss = test_metrics.loss
                    min_test_loss = min(min_test_loss, test_loss)

                    if (test_loss == min_test_loss) or (epoch % CHECKPOINT_FREQUENCY == 0):
                        ckpt = create_ckpt(model=vae,
                                           optimizer=None,
                                           epoch=epoch,
                                           hyperparams_dict=params,
                                           history_dict=history_dict)
                        #log_object_as_artifact(name="last_ckpt", obj=ckpt)  # log file into neptune
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

# # Check segmentation WITH tiling
img_to_segment = train_loader.img[0, :, 1000:1300, 2100:2400]
roi_mask_to_segment = train_loader.roi_mask[0, :, 1000:1300, 2100:2400]
tiling = vae.segment_with_tiling(single_img=img_to_segment,
                                 roi_mask=roi_mask_to_segment,
                                 crop_size=None,
                                 stride=(40, 40),
                                 n_objects_max_per_patch=None,
                                 prob_corr_factor=None,
                                 overlap_threshold=None,
                                 radius_nn=10,
                                 batch_size=64)
log_object_as_artifact(name="tiling", obj=tiling, verbose=True)
tiling_fig = plot_tiling(tiling)
exp.stop()

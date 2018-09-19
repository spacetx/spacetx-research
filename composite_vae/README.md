# Composite Variational Auto-Encoder

We approach the problem of unsupervised image segmentation from the perspective of deep generative models. In particular, we explore a novel extension of VAE which we call _Composite Variational Auto-Encoder (comp-VAE)_ to that end.

This repository contains prototypes and explorations of comp-VAE and its applicability to cell segmentation tasks (for the time being, only the DAPI channel).

### Description of Files

**synth_dapi_stain_generator.ipynb**: This notebook generates synthetic images of DAPI stains. The images could be used to train a variational auto-encoder (VAE) for testing purposes (e.g. to generate not-too-ugly images while testing and implementing comp-VAE).

**synth_dapi_stain_vae.ipynb**: This notebook trains a VAE on a dataset of DAPI stains. The decoder can be plugged in the comp-VAE model for testing purposes.

**composite_vae_proto.ipynb**: This notebook provides and explores comp-VAE. At the moment, only the generator is implemented.

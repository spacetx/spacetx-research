#!/usr/bin/env python
# coding: utf-8

import neptune
from MODULES.utilities import flatten_dict

params = load_json_as_dict("./ML_parameters.json")

neptune.set_project(params["neptune_project"])

exp: neptune.experiments.Experiment = \
    neptune.create_experiment(params=flatten_dict(params),
                              upload_source_files=["./main_mnist.py", "./ML_parameters.json"],
                              upload_stdout=True,
                              upload_stderr=True)

for delta_epoch in range(1, NUM_EPOCHS+1):
    epoch = delta_epoch+epoch_restart    

    vae.prob_corr_factor = linear_interpolation(epoch,
                                                values=params["shortcut_prob_corr_factor"]["values"],
                                                times=params["shortcut_prob_corr_factor"]["times"])
    exp.log_metric("prob_corr_factor", vae.prob_corr_factor)
        

exp.stop()

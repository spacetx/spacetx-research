import neptune
import torch.nn
import numpy
import matplotlib.figure
from typing import Optional, List
from MODULES.utilities import save_obj
from neptunecontrib.api import log_chart
from MODULES.namedtuple import ConcordanceIntMask


def log_img_only(name: str,
                 fig: matplotlib.figure.Figure,
                 experiment:  Optional[neptune.experiments.Experiment] = None,
                 verbose: bool = False):
    if verbose:
        print("inside log_img_only -> "+name)
    _exp = experiment if experiment else neptune
    _exp.log_image(name, fig)
    if verbose:
        print("leaving log_img_only -> "+name)


def log_img_and_chart(name: str,
                      fig: matplotlib.figure.Figure,
                      experiment:  Optional[neptune.experiments.Experiment] = None,
                      verbose: bool = False):
    if verbose:
        print("inside log_img_and_chart -> "+name)
    _exp = experiment if experiment else neptune
    log_chart(name, fig, _exp)
    _exp.log_image(name, fig)
    if verbose:
        print("leaving log_img_and_chart -> "+name)


def log_model_summary(model: torch.nn.Module,
                      experiment: Optional[neptune.experiments.Experiment] = None,
                      verbose: bool = False):
    if verbose:
        print("inside log_model_summary")

    _exp = experiment if experiment else neptune

    for x in model.__str__().split('\n'):
        # replace leading spaces with '-' character
        n = len(x) - len(x.lstrip(' '))
        _exp.log_text("model summary", '-' * n + x)

    if verbose:
        print("leaving log_model_summary")


def log_object_as_artifact(name: str,
                           obj: object,
                           experiment: Optional[neptune.experiments.Experiment] = None,
                           verbose: bool = False):
    if verbose:
        print("inside log_object_as_artifact")

    path = name+".pt"
    save_obj(obj=obj, path=path)
    _exp = experiment if experiment else neptune
    _exp.log_artifact(path)

    if verbose:
        print("leaving log_object_as_artifact")


def log_matplotlib_as_png(name: str,
                          fig: matplotlib.figure.Figure,
                          experiment: Optional[neptune.experiments.Experiment] = None,
                          verbose: bool = False):
    if verbose:
        print("log_matplotlib_as_png")

    _exp = experiment if experiment else neptune
    fig.savefig(name+".png")  # save to local file
    _exp.log_image(name, name+".png")  # log file to neptune

    if verbose:
        print("leaving log_matplotlib_as_png")


def log_dict_metrics(metrics: dict,
                     prefix: str = "",
                     keys_exclude: List[str] = [""],
                     experiment: Optional[neptune.experiments.Experiment] = None,
                     verbose: bool = False):
    if verbose:
        print("inside log_dict_metrics")

    _exp = experiment if experiment else neptune

    for key, value in metrics.items():
        if key in keys_exclude:
            continue

        if isinstance(value, float):
            _exp.log_metric(prefix + key, value)
        elif isinstance(value, numpy.ndarray):
            for i, x in enumerate(value):
                _exp.log_metric(prefix + key + "_" + str(i), x)
        elif isinstance(value, torch.Tensor):
            for i, x in enumerate(value):
                _exp.log_metric(prefix + key + "_" + str(i), x.item())
        else:
            print(key)
            print(type(value), value)
            raise Exception

    if verbose:
        print("leaving log_dict_metrics")


def log_concordance(concordance: ConcordanceIntMask,
                    prefix: str = "",
                    experiment: Optional[neptune.experiments.Experiment] = None,
                    verbose: bool = False):
    if verbose:
        print("inside log_concordance")

    tmp_dict = {"iou": concordance.iou,
                "mutual_information": concordance.mutual_information,
                "intersection": concordance.intersection_mask.sum().item(),
                "delta_n": concordance.delta_n,
                "matching_instances": concordance.n_reversible_instances}

    log_dict_metrics(metrics=tmp_dict, prefix=prefix, experiment=experiment)

    if verbose:
        print("leaving log_concordance")


def log_last_ckpt(name: str,
                  ckpt: dict,
                  experiment: Optional[neptune.experiments.Experiment] = None,
                  verbose: bool = False,
                  delete_previous_ckpt: bool = True):

    if verbose:
        print("inside log_last_ckpt")

    _exp = experiment if experiment else neptune
    path = name+".pt"
    #if delete_previous_ckpt:
    #    _exp.delete_artifacts(path)
    save_obj(obj=ckpt, path=path)
    print("logging artifact")
    _exp.log_artifact(path)

    if verbose:
        print("leaving log_last_ckpt")

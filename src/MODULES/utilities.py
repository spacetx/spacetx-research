import json
import torch
import numpy
import dill
from typing import Union, Optional
from MODULES.namedtuple import BB, ConcordanceIntMask
import skimage.measure


def convert_to_box_list(x: torch.Tensor) -> torch.Tensor:
    """ takes input of shape: (batch, ch, width, height)
        and returns output of shape: (n_list, batch, ch)
        where n_list = width x height
    """
    assert len(x.shape) == 4
    batch_size, ch, width, height = x.shape
    return x.permute(2, 3, 0, 1).view(width*height, batch_size, ch)


def invert_convert_to_box_list(x: torch.Tensor, original_width: int, original_height: int) -> torch.Tensor:
    """ takes input of shape: (width x height, batch, ch)
        and return shape: (batch, ch, width, height)
    """
    assert len(x.shape) == 3
    n_list, batch_size, ch = x.shape
    assert n_list == original_width * original_height
    return x.permute(1, 2, 0).view(batch_size, ch, original_width, original_height)


def tmaps_to_bb(tmaps, width_raw_image: int, height_raw_image: int, min_box_size: float, max_box_size: float):
    tx_map, ty_map, tw_map, th_map = torch.split(tmaps, 1, dim=-3)
    n_width, n_height = tx_map.shape[-2:]
    ix_array = torch.arange(start=0, end=n_width, dtype=tx_map.dtype, device=tx_map.device)
    iy_array = torch.arange(start=0, end=n_height, dtype=tx_map.dtype, device=tx_map.device)
    ix_grid, iy_grid = torch.meshgrid([ix_array, iy_array])

    bx_map: torch.Tensor = width_raw_image * (ix_grid + tx_map) / n_width
    by_map: torch.Tensor = height_raw_image * (iy_grid + ty_map) / n_height
    bw_map: torch.Tensor = min_box_size + (max_box_size - min_box_size) * tw_map
    bh_map: torch.Tensor = min_box_size + (max_box_size - min_box_size) * th_map
    return BB(bx=convert_to_box_list(bx_map).squeeze(-1),
              by=convert_to_box_list(by_map).squeeze(-1),
              bw=convert_to_box_list(bw_map).squeeze(-1),
              bh=convert_to_box_list(bh_map).squeeze(-1))


def linear_interpolation(t: Union[numpy.array, float], values: tuple, times: tuple) -> Union[numpy.array, float]:
    """ Makes an interpolation between (t_in,v_in) and (t_fin,v_fin)
        For time t>t_fin and t<t_in the value of v is clamped to either v_in or v_fin
        Usage:
        epoch = numpy.arange(0,100,1)
        v = linear_interpolation(epoch, values=[0.0,0.5], times=[20,40])
        plt.plot(epoch,v)
    """
    v_in, v_fin = values  # initial and final values
    t_in, t_fin = times   # initial and final times

    if t_fin >= t_in:
        den = max(t_fin-t_in, 1E-8)
        m = (v_fin-v_in)/den
        v = v_in + m*(t-t_in)
    else:
        raise Exception("t_fin should be greater than t_in")

    v_min = min(v_in, v_fin)
    v_max = max(v_in, v_fin)
    return numpy.clip(v, v_min, v_max)


def QC_on_integer_mask(integer_mask: Union[torch.Tensor, numpy.ndarray], min_area):
    """ This function filter the labels by some criteria (for example by min size).
        Add more QC in the future"""
    if isinstance(integer_mask, torch.Tensor):
        labels = skimage.measure.label(integer_mask.cpu(), background=0, return_num=False, connectivity=2)
    else:
        labels = skimage.measure.label(integer_mask, background=0, return_num=False, connectivity=2)

    mydict = skimage.measure.regionprops_table(labels, properties=['label', 'area'])
    my_filter = mydict["area"] > min_area

    bad_labels = mydict["label"][~my_filter]
    old2new = numpy.arange(mydict["label"][-1] + 1)
    old2new[bad_labels] = 0
    new_integer_mask = old2new[labels]

    if isinstance(integer_mask, torch.Tensor):
        return torch.from_numpy(new_integer_mask).to(dtype=integer_mask.dtype, device=integer_mask.device)
    else:
        return new_integer_mask.astype(integer_mask.dtype)


def remove_label_gaps(label: torch.Tensor):
    assert torch.is_tensor(label)
    assert len(label.shape) == 2
    count = torch.bincount(label.flatten())
    exist = (count > 0).to(label.dtype)
    old2new = torch.cumsum(exist, dim=-1) - exist[0]
    print(old2new)
    return old2new[label.long()]
    

def concordance_integer_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> ConcordanceIntMask:
    """ Compute measure of concordance between two partitions:
        joint_distribution
        mutual_information
        delta_n
        iou

        We use the peaks of the join distribution to extract the mapping between membership labels.
    """
    assert torch.is_tensor(mask1)
    assert torch.is_tensor(mask2)
    assert mask1.shape == mask2.shape
    assert mask1.device == mask2.device

    nx = torch.max(mask1).item()  # 0,1,..,nx
    ny = torch.max(mask2).item()  # 0,1,..,ny
    z = mask2 + mask1 * (ny + 1)  # 0,1,...., nx*ny
    count_z = torch.bincount(z.flatten(),
                             minlength=(nx+1)*(ny+1)).float().view(nx+1, ny+1).to(dtype=torch.float,
                                                                                  device=mask1.device)
    # Now remove the background
    pxy = count_z.clone()
    pxy[0, :] = 0
    pxy[:, 0] = 0

    # computing the mutual information
    pxy /= torch.sum(pxy)
    px = torch.sum(pxy, dim=-1)
    py = torch.sum(pxy, dim=-2)
    term_xy = pxy * torch.log(pxy)
    term_x = px * torch.log(px)
    term_y = py * torch.log(py)
    mutual_information = term_xy[pxy > 0].sum() - term_x[px > 0].sum() - term_y[py > 0].sum()

    # Extract the most likely mappings
    mask1_to_mask2 = torch.max(pxy, dim=-1)[1]
    mask2_to_mask1 = torch.max(pxy, dim=-2)[1]
    assert mask1_to_mask2.shape[0] == nx + 1
    assert mask2_to_mask1.shape[0] == ny + 1

    # Find one-to-one correspondence among IDs
    id_mask1 = torch.arange(nx + 1, device=mask1.device, dtype=torch.long)  # [0,1,.., nx]
    is_id_reversible = (mask2_to_mask1[mask1_to_mask2[id_mask1]] == id_mask1)
    n_reversible_instances = is_id_reversible[1:].sum().item()  # exclude id=0 bc background

    # Find one-to-one correspondence among pixels
    intersection_mask = (mask1_to_mask2[mask1] == mask2) * (mask2_to_mask1[mask2] == mask1) * (mask1 > 0) * (mask2 > 0)
    intersection = intersection_mask.sum()
    union = torch.sum(mask1 > 0) + torch.sum(mask2 > 0) - intersection  # exclude background
    iou = intersection.float() / union

    return ConcordanceIntMask(intersection_mask=intersection_mask,
                              joint_distribution=pxy,
                              mutual_information=mutual_information.item(),
                              delta_n=ny - nx,
                              iou=iou.item(),
                              n_reversible_instances=n_reversible_instances)


def flatten_list(ll):
    if not ll:  # equivalent to if ll == []
        return ll
    elif isinstance(ll[0], list):
        return flatten_list(ll[0]) + flatten_list(ll[1:])
    else:
        return ll[:1] + flatten_list(ll[1:])


def flatten_dict(dd, separator='_', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}


def save_obj(obj, path):
    with open(path, 'wb') as f:
        torch.save(obj, f,
                   pickle_module=dill,
                   pickle_protocol=2,
                   _use_new_zipfile_serialization=True)


def load_obj(path):
    with open(path, 'rb') as f:
        return torch.load(f, pickle_module=dill)


def load_json_as_dict(path):
    with open(path, 'rb') as f:
        return json.load(f)


def save_dict_as_json(my_dict, path):
    with open(path, 'w') as f:
        return json.dump(my_dict, f)


def roller_2d(a: torch.tensor, b: Optional[torch.tensor] = None, radius: int = 2):
    """ Performs rolling of the last two spatial dimensions.
        For each point consider half a square. Each pair of points will appear once.
        Number of channels: [(2r+1)**2 - 1]/2
        For example for a radius = 2 the full square is 5x5. The number of pairs is: 12
    """
    dxdy_list = []
    for dx in range(0, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy <= 0:
                continue
            dxdy_list.append((dx, dy))

    for dxdy in dxdy_list:
        a_tmp = torch.roll(torch.roll(a, dxdy[0], dims=-2), dxdy[1], dims=-1)
        b_tmp = None if b is None else torch.roll(torch.roll(b, dxdy[0], dims=-2), dxdy[1], dims=-1)
        yield a_tmp, b_tmp


def append_to_dict(source: Union[tuple, dict],
                   destination: dict,
                   prefix_include: str = None,
                   prefix_exclude: str = None,
                   prefix_to_add: str = None):
    """ Use typing.
        For now: prefix_include is str or tuple of str
        For now: prefix_exclude is str or tuple of str
        For now: prefix_to_add is str """

    def _get_x_y(_key, _value):
        if (prefix_include is None or _key.startswith(prefix_include)) and (prefix_exclude is None or
                                                                            not _key.startswith(prefix_exclude)):
            x = _key if prefix_to_add is None else prefix_to_add + _key
            try:
                y = _value.item()
            except (AttributeError, ValueError):
                y = _value
            return x, y
        else:
            return None, None

    if isinstance(source, dict):
        for key, value in source.items():
            x, y = _get_x_y(key, value)
            if x is not None:
                destination[x] = destination.get(x, []) + [y]

    elif isinstance(source, tuple):
        for key in source._fields:
            value = getattr(source, key)
            x, y = _get_x_y(key, value)
            if x is not None:
                destination[x] = destination.get(x, []) + [y]

    else:
        print(type(source))
        raise Exception

    return destination


def compute_ranking(x: torch.Tensor) -> torch.Tensor:
    """ Given a vector of shape: n, batch_size
        For each batch dimension it ranks the n elements"""
    assert len(x.shape) == 2
    n, batch_size = x.shape
    _, order = torch.sort(x, dim=-2, descending=False)

    # this is the fast way which uses indexing on the left
    rank = torch.zeros_like(order)
    batch_index = torch.arange(batch_size, dtype=order.dtype, device=order.device).view(1, -1).expand(n, batch_size)
    rank[order, batch_index] = torch.arange(n, dtype=order.dtype, device=order.device).view(-1, 1).expand(n, batch_size)
    return rank


def compute_average_in_box(imgs: torch.Tensor, bounding_box: BB) -> torch.Tensor:
    """ Input batch of images: batch_size x ch x w x h
        z_where collections of [bx,by,bw,bh]
        bx.shape = batch x n_box
        similarly for by,bw,bh
        Output:
        av_intensity = n_box x batch_size
    """
    # cumulative sum in width and height, standard sum in channels
    cum_sum = torch.cumsum(torch.cumsum(imgs.sum(dim=-3), dim=-1), dim=-2)
    assert len(cum_sum.shape) == 3
    batch_size, w, h = cum_sum.shape

    # compute the x1,y1,x3,y3
    x1 = (bounding_box.bx - 0.5 * bounding_box.bw).long().clamp(min=0, max=w)
    x3 = (bounding_box.bx + 0.5 * bounding_box.bw).long().clamp(min=0, max=w)
    y1 = (bounding_box.by - 0.5 * bounding_box.bh).long().clamp(min=0, max=h)
    y3 = (bounding_box.by + 0.5 * bounding_box.bh).long().clamp(min=0, max=h)
    assert x1.shape == x3.shape == y1.shape == y3.shape  # n_boxes, batch_size

    # compute the area
    # Note that this way penalizes boxes that go out-of-bound
    # This is in contrast to area = (x3-x1)*(y3-y1) which does NOT penalize boxes out of bound
    area = bounding_box.bw * bounding_box.bh
    assert area.shape == x1.shape == x3.shape == y1.shape == y3.shape
    n_boxes, batch_size = area.shape

    # compute the total intensity in each box
    b_index = torch.arange(start=0, end=batch_size, step=1, device=x1.device,
                           dtype=x1.dtype).view(1, -1).expand(n_boxes, -1)
    assert b_index.shape == x1.shape

    x1_ge_1 = (x1 >= 1).float()
    x3_ge_1 = (x3 >= 1).float()
    y1_ge_1 = (y1 >= 1).float()
    y3_ge_1 = (y3 >= 1).float()
    tot_intensity = cum_sum[b_index, x3-1, y3-1] * x3_ge_1 * y3_ge_1 + \
                    cum_sum[b_index, x1-1, y1-1] * x1_ge_1 * y1_ge_1 - \
                    cum_sum[b_index, x1-1, y3-1] * x1_ge_1 * y3_ge_1 - \
                    cum_sum[b_index, x3-1, y1-1] * x3_ge_1 * y1_ge_1
    return tot_intensity / area


def sample_from_constraints_dict(dict_soft_constraints: dict,
                                 var_name: str,
                                 var_value: torch.Tensor,
                                 verbose: bool = False,
                                 chosen: Optional[int] = None) -> torch.Tensor:

    cost = torch.zeros_like(var_value)
    var_constraint_params = dict_soft_constraints[var_name]

    if 'lower_bound_value' in var_constraint_params:
        left = var_constraint_params['lower_bound_value']
        width_low = var_constraint_params['lower_bound_width']
        exponent_low = var_constraint_params['lower_bound_exponent']
        strength_low = var_constraint_params['lower_bound_strength']
        activity_low = torch.clamp(left + width_low - var_value, min=0.) / width_low
        cost += strength_low * activity_low.pow(exponent_low)

    if 'upper_bound_value' in var_constraint_params:
        right = var_constraint_params['upper_bound_value']
        width_up = var_constraint_params['upper_bound_width']
        exponent_up = var_constraint_params['upper_bound_exponent']
        strength_up = var_constraint_params['upper_bound_strength']
        activity_up = torch.clamp(var_value - right + width_up, min=0.) / width_up
        cost += strength_up * activity_up.pow(exponent_up)

    if 'strength' in var_constraint_params:
        strength = var_constraint_params['strength']
        exponent = var_constraint_params['exponent']
        cost += strength * var_value.pow(exponent)

    if verbose:
        if chosen is None:
            print("constraint name ->", var_name)
            print("input value ->", var_value)
            print("cost ->", cost)
        else:
            print("constraint name ->", var_name)
            print("input value ->", var_value[..., chosen])
            print("cost ->", cost[..., chosen])

    return cost

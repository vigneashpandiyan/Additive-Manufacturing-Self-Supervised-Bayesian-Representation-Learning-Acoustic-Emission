import numpy as np
from tqdm import tqdm
# import Utils.helper as hlp



def cut_piece2C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len/(2*2)

    if perc<1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len-win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1-start2)<(win_class):
        label=0
    else:
        label=1
   
    return ts[start1:end1, ...], ts[start2:end2, ...], label


def cut_piece3C(ts, perc=.1):
    seq_len = ts.shape[0]
    win_class = seq_len/(2*3)

    if perc<1:
        win_len = int(perc * seq_len)
    else:
        win_len = perc

    start1 = np.random.randint(0, seq_len-win_len)
    end1 = start1 + win_len
    start2 = np.random.randint(0, seq_len - win_len)
    end2 = start2 + win_len

    if abs(start1-start2)<(win_class):
        label=0
    elif abs(start1-start2)<(2*win_class):
        label=1
    else:
        label=2
    
    return ts[start1:end1, ...], ts[start2:end2, ...], label



def slidewindow(ts, horizon=.2, stride=0.2):
    xf = []
    yf = []
    for i in range(0, ts.shape[0], int(stride * ts.shape[0])):
        horizon1 = int(horizon * ts.shape[0])
        if (i + horizon1 + horizon1 <= ts.shape[0]):
            xf.append(ts[i:i + horizon1,0])
            yf.append(ts[i + horizon1:i + horizon1 + horizon1, 0])

    xf = np.asarray(xf)
    yf = np.asarray(yf)

    return xf, yf



def cutout(ts, perc=.1):
    seq_len = ts.shape[0]
    new_ts = ts.copy()
    win_len = int(perc * seq_len)
    start = np.random.randint(0, seq_len-win_len-1)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    # print("[INFO] start={}, end={}".format(start, end))
    new_ts[start:end, ...] = 0
    # return new_ts, ts[start:end, ...]
    return new_ts




def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(1, x.shape[1]))
    x_ = np.multiply(x, factor[:, :])

  
    return x_

def magnitude_warp(x, sigma=0.2, knot=4, plot=False):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(1, knot + 2, x.shape[1]))
    warp_steps = (np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1., num=knot + 2))).T

    li = []
    for dim in range(x.shape[1]):
        li.append(CubicSpline(warp_steps[:, dim], random_warps[0, :, dim])(orig_steps))
    warper = np.array(li).T

    x_ = x * warper

   
    return x_


def time_warp(x, sigma=0.2, knot=4, plot=False):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[0])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(1, knot + 2, x.shape[1]))
    warp_steps = (np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for dim in range(x.shape[1]):
        time_warp = CubicSpline(warp_steps[:, dim],
                                warp_steps[:, dim] * random_warps[0, :, dim])(orig_steps)
        scale = (x.shape[0] - 1) / time_warp[-1]
        ret[:, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[0] - 1),
                                   x[:, dim]).T
    
    return ret


def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio * x.shape[0]).astype(int)
    if target_len >= x.shape[0]:
        return x
    starts = np.random.randint(low=0, high=x.shape[0] - target_len, size=(1)).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for dim in range(x.shape[1]):
        ret[:, dim] = np.interp(np.linspace(0, target_len, num=x.shape[0]), np.arange(target_len),
                                   x[starts[0]:ends[0], dim]).T
    return ret



def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, 1)
    warp_size = np.ceil(window_ratio * x.shape[0]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(low=1, high=x.shape[0] - warp_size - 1, size=(1)).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    pat=x
    for dim in range(x.shape[1]):
        start_seg = pat[:window_starts[0], dim]
        window_seg = np.interp(np.linspace(0, warp_size - 1,
                                           num=int(warp_size * warp_scales[0])), window_steps,
                               pat[window_starts[0]:window_ends[0], dim])
        end_seg = pat[window_ends[0]:, dim]
        warped = np.concatenate((start_seg, window_seg, end_seg))
        ret[:, dim] = np.interp(np.arange(x.shape[0]), np.linspace(0, x.shape[0] - 1., num=warped.size),
                                   warped).T
    return ret


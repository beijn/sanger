import matplotlib.pyplot as plt
import numpy as np

import itertools as it
from statistics import mean

try: 
  f = plt.figure()
  plt.close(f)
except Exception as e:
  print(f'NOTE: cellnet.plot will try switching to headless plotting, because {e.__class__.__name__}: {e}')
  import matplotlib
  matplotlib.use("Agg")


ZOOM = 1
def set_zoom(zoom): 
  global ZOOM
  ZOOM = zoom


import pathlib
def imgid(path): return pathlib.Path(path).stem


def heatmap(hm, ax=None, alpha=lambda value: value, color='#ff0000'):
  """Overlay heatmap on image. Color cam be color or color map. Alpha: map value to transparency."""
  from matplotlib.colors import Colormap, ListedColormap 

  if hm.ndim == 4 and hm.shape[0] == 1: hm = hm[0]
  if hm.ndim == 3 and hm.shape[-1] == 1: hm = hm[:,:,0]
  elif hm.ndim == 3 and hm.shape[ 0] == 1: hm = hm[0,:,:]
  else: assert hm.ndim == 2, f"Expected 2D heatmap, got shape {hm.shape}."

  hm = hm.astype(np.float32)
  hm = ((hm - hm.min()) / (1e-9+ hm.max() - hm.min()) * 255).astype(np.uint8)

  alpha = np.vectorize(alpha)(np.linspace(0, 1, 256))
  if not isinstance(color, Colormap): color = ListedColormap(color)

  out = color(hm); out[:,:,3] = alpha[hm]
  out[0,0,3] = 1; out[0,1,3]=0  # scale alpha 
  return image(out, ax=ax)


def points(ax, points, labels=None, radius=10.0, colormap={1: 'black', 2: '#7700ff'}, marker='o', **scatter_args):
  """"[(x,y), ...]"""
  s = np.pi*radius**2*10000 if marker == 'o' else 10000*radius**2
  if 'lw' in scatter_args: scatter_args['lw']*=100
  cs = colormap if labels is None else [colormap[i] for i in labels] if type(colormap) is not str else colormap
  fc = 'none' if marker == 'o' else cs
  ax.scatter(*zip(*points), **dict(facecolors=fc, edgecolors=cs, marker=marker, alpha=0.6, s=s, lw=200) | scatter_args)
  

def grid(grid, shape, zoom=None):
  global ZOOM
  if zoom is None: zoom = ZOOM
  w,h = grid
  fig = plt.figure(frameon=False, layout='tight', dpi=1)
  fig.set_size_inches(shape[0]/fig.dpi*zoom*w, shape[1]/fig.dpi*zoom*h) 

  axs = [fig.add_axes((i/w, j/h, 1/w, 1/h)) for i,j in it.product(range(w), range(h))]
  for ax in axs: ax.axis('off')

  return fig, axs


def image(img, ax=None, zoom=None, norm=True, **imshow_kwargs):
  global ZOOM
  if zoom is None: zoom = ZOOM
  if img.ndim == 3 and img.shape[0] in (1,3,4): img = np.transpose(img, (1,2,0))
  assert img.ndim == 2 or (img.ndim == 3 and (img.shape[0] in (1,3,4) or img.shape[-1] in (1,3,4))), \
    f"Plot only 2D gray or RGB(A)-channel-last images. Got shape {img.shape}."

  if ax == None:
    # no whitespace and 1:1 pixel resolution
    fig = plt.figure(frameon=False, layout='tight', dpi=1)
    ax = fig.add_axes((0,0,1,1))
    fig.set_size_inches(img.shape[1]/fig.dpi*zoom, img.shape[0]/fig.dpi*zoom)  
    ax.axis('off')

  if norm: img = (img - img.min()) / (1e-9+ img.max() - img.min())  # NOTE: else user has to ensure that image is in [0,1]
  elif (max:=img.max()) > 1 and max <= 255 and img.min() >= 0: img /= 255
  ax.imshow(img, interpolation='none', **imshow_kwargs)
  return ax


def overlay(x, y=None, m=None, k=None, l=None, sigma=5.0, ax=None, args_point={}, args_image={}):
  ax = image(x, ax=ax, **args_image)
  if m is not None: heatmap(1-m, ax=ax, alpha=lambda x: 0.5*x, color='#000000')
  if y is not None: heatmap(y, ax=ax, alpha=lambda x: 1.0*x, color='#ff0000')
  if k is not None and l is not None and len(k) and len(l): points(ax, k, l, sigma*1.5, **args_point)
  return ax

def diff(y, z, m=None, k=None, l=None, sigma=4.0, ax=None):
  title = f"Difference between Target and Predicted Heatmap"
  D = y-z; D[0, 1,0] = -1; D[0, 1,1] = 1 
  ax = image(D, ax=ax, cmap='coolwarm')
  if m is not None: heatmap(1-m, ax=ax, alpha=lambda x: 0.2*x, color='#000000')
  if k is not None and l is not None: points(ax, k, l, sigma*1.5)
  return ax


def save(ax, path, transparent=False):
  ax.figure.savefig(path, transparent=transparent, pil_kwargs=dict(compress_level=9))


# TODO use seaborn and overlay KDE to easier see progress despite noise
def train_graph(epochs, log, keys=None, clear=False, info={}, key2text={}, accuracy=True, vi=[], **unknown):
  if clear: 
    from IPython.display import clear_output
    clear_output(wait=True)
  log['lr'] /= log['lr'].max()    

  _, axs = plt.subplots(_r := 2 if accuracy else 1,1, figsize=(10,7.5*_r))
  if not accuracy: axs = [axs]

  for ax, T in zip(axs, ['Loss', 'Accuracy']):
    title = f"Training {T}\nvi: {', '.join([imgid(i) for i in vi])}. {', '.join([f'{key2text.get(k, k)}: {v}' for k,v in [('e', epochs), *info.items()]])}"
    ax.set_title(title)
    print(title)

  for key in (log if keys is None else keys):
    if key in "lr tl vl".split(' '): ax = axs[0]
    elif not accuracy: continue
    else: ax = axs[1]
    v = log['key'].apply(mean)   if hasattr(log[key][0], '__len__') else log[key]
    ax.plot(v, label=key2text[key] if key2text else key)

  for ax in axs:
    ax.set_yscale('log')
    ax.legend(loc='upper right')
  plt.show()


def regplot(data, dim, key2text, remove_outliers_below:None|float=0, sort_by='va_mean'):
  import seaborn as sns
  from cellnet.data import imgid
  if sort_by is not None: data = data.sort_values(by=[sort_by], ascending=True).reset_index(drop=True)
  data = data.explode(keys := ['ta', 'va', 'tl', 'vl'] )
  for key in keys: data[key] = data[key].apply(lambda x: float(x) if x != 'nan' else float('nan'))
  by_val_img = data[key2text['vi']].apply(lambda vi: 
    (_vi := [imgid(i) for i in vi], _vi[0] if len(_vi)==1 else ', '.join(_vi))[-1])
  if remove_outliers_below is not None: 
    for key in keys: 
      fun = lambda dimval, vi, x: (print(f"NOTE: Outlier hidden at {dim}={dimval}, {key} = {x}"+(f" (vi: {vi})")), remove_outliers_below)[-1] if x < remove_outliers_below else x
      data[key] = data.apply(lambda row: fun(row[dim], row[key2text['vi']], row[key]), axis=1)

  axs = [plt.subplots(1,1, figsize=(5*(golden := (1 + 5 ** 0.5) / 2),5))[1] for _ in range(len(keys))]
  for ax, key in zip(axs, keys):
    try: 
      if len(data[dim].unique()) <= 1: raise Exception("Only one unique x-value.")
      sns.regplot(x=dim, y=key, data=data, scatter=False, ax=ax) 
      sns.scatterplot(ax=ax, data=data, x=dim, y=key, hue=by_val_img)
    except Exception as e: 
      print(f"Log. Cannot plot regression {dim}-{key}, likely because the variables are categorical and not numerical. ({e})") 
      sns.boxplot(x=dim, y=key, data=data, ax=ax, orient='v', fill=False)
      sns.swarmplot(ax=ax, data=data, x=dim, y=key, hue=by_val_img)
    dimtext = key2text[dim] if dim in key2text else f'{dim}'
    ax.set_title(f'{key2text[key]} vs {dimtext}')
    ax.set_xlabel(dimtext)
    ax.set_ylabel(key2text[key])
    if by_val_img.nunique() > 1: sns.move_legend(ax, "lower left")
    # make the x-axis text readable
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

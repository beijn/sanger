from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import itertools as it
import seaborn as sns

# this code assumes HWC

ZOOM = 1
def set_zoom(zoom): global ZOOM; ZOOM = zoom


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


def points(ax, points, labels=None, radius=7, colormap={1: 'black', 2: '#7700ff'}, marker='o', **scatter_args):
  """"[(x,y), ...]"""
  s = np.pi*radius**2*10000 if marker == 'o' else 10000*radius**2
  if 'lw' in scatter_args: scatter_args['lw']*=100
  cs = 'black' if labels is None else [colormap[i] for i in labels]
  ax.scatter(*zip(*points), **dict(facecolors='none', edgecolors=cs, marker=marker, alpha=0.4, s=s, lw=150) | scatter_args)
  

def grid(grid, shape, zoom=None):
  global ZOOM
  if zoom is None: zoom = ZOOM
  w,h = grid
  fig = plt.figure(frameon=False, layout='tight', dpi=1)
  fig.set_size_inches(shape[0]/fig.dpi*zoom*w, shape[1]/fig.dpi*zoom*h) 

  axs = [fig.add_axes((i/w, j/h, 1/w, 1/h)) for i,j in it.product(range(w), range(h))]
  for ax in axs: ax.axis('off')

  return fig, axs


def image(img, ax=None, zoom=None, exact=True, **imshow_kwargs):
  global ZOOM
  if zoom is None: zoom = ZOOM
  if img.ndim == 3 and img.shape[0] in (1,3,4): img = np.transpose(img, (1,2,0))
  assert img.ndim == 2 or (img.ndim == 3 and (img.shape[0] in (1,3,4) or img.shape[-1] in (1,3,4))), \
    f"Plot only 2D gray or RGB(A)-channel-last images. Got shape {img.shape}."

  if ax == None:
    # no whitespace and 1:1 pixel resolution
    fig = plt.figure(frameon=not exact, layout='tight', dpi=1)
    ax = fig.add_axes((0,0,1,1))
    fig.set_size_inches(img.shape[1]/fig.dpi*zoom, img.shape[0]/fig.dpi*zoom)  
    ax.axis('off')

  img = (img - img.min()) / (1e-9+ img.max() - img.min())
  ax.imshow(img, interpolation='none', **imshow_kwargs)
  return ax


def overlay(x, y=None, m=None, k=None, l=None, sigma=4, ax=None):
  ax = image(x, ax=ax)
  if m is not None: heatmap(1-m, ax=ax, alpha=lambda x: 0.5*x, color='#000000')
  if y is not None: heatmap(y, ax=ax, alpha=lambda x: 1.0*x, color='#ff0000')
  if k is not None and l is not None: points(ax, k, l, sigma*2)
  return ax

def diff(y, z, m=None, k=None, l=None, sigma=4, ax=None):
  title = f"Difference between Target and Predicted Heatmap"
  D = y-z; D[0, 1,0] = -1; D[0, 1,1] = 1 
  ax = image(D, ax=ax, cmap='coolwarm')
  if m is not None: heatmap(1-m, ax=ax, alpha=lambda x: 0.2*x, color='#000000')
  if k is not None and l is not None: points(ax, k, l, sigma*2)
  return ax


def save(ax, path, transparent=False):
  ax.figure.savefig(path, transparent=transparent, pil_kwargs=dict(compress_level=9))


def train_graph(epochs, log, keys=None, clear=False, info={}, key2text={}, accuracy=True, **unknown):
  if clear: clear_output(wait=True)
  log['lr'] /= log['lr'].max()    

  _, axs = plt.subplots(_r := 2 if accuracy else 1,1, figsize=(10,7.5*_r))
  if not accuracy: axs = [axs]

  for ax, T in zip(axs, ['Loss', 'Accuracy']):
    ax.set_title(f"Training {T}\n{', '.join([f'{key2text[k]}: {v}' for k,v in [('e', epochs), *info.items()]])}")

  for key in (log if keys is None else keys):
    if key in "lr tl vl".split(' '): ax = axs[0]
    elif not accuracy: continue
    else: ax = axs[1]
    ax.plot(log[key], label=key2text[key] if key2text else key)

  for ax in axs:
    ax.set_yscale('log')
    ax.legend(loc='upper right')
  plt.show()


def regplot(R, dim, key2text):
  vi=key2text['vi']
  
  fig, axs = plt.subplots(2,2, figsize=(15,10))
  for ax, (key, text) in zip(axs.flat, key2text.items()):
    if key in "ta va tl vl".split(' '):
      ax = sns.scatterplot(ax=ax, data=R, 
        x=dim, y=key, hue=R[vi].map(lambda l: l[0])) 
      sns.regplot(x=dim, y=key, data=R, scatter=False, ax=ax) 
      ax.set_title(f'{text} vs {key2text[dim]}')
      ax.set_xlabel(key2text[dim])
      ax.set_ylabel(key2text[key])
      
      sns.move_legend(ax, "lower left")

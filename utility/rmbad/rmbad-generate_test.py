import os; os.chdir('../..')

# %% # Definitions
from types import SimpleNamespace as obj

import cellnet.data as data
import cellnet.plot as plot

import albumentations as A
from albumentations.pytorch import ToTensorV2

import json
import numpy as np
import matplotlib.pyplot as plt

transforms = A.Compose(transforms=[
    A.Normalize(normalization='image_per_channel'),
    ToTensorV2(transpose_mask=True, always_apply=True)], 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True) 
  )

b = data.batch2cpu(next(iter(data.mk_loader(['data/images/1.jpg'], 1, transforms, cfg=obj(sigma=5.0, maxdist=24)))))[0] 

p2L = np.load('p2L-1.npy')

def image(img, ax=None, zoom=1, exact=True, **imshow_kwargs):
  if img.ndim == 3 and img.shape[0] in (1,3,4): img = np.transpose(img, (1,2,0))
  assert img.ndim == 2 or (img.ndim == 3 and (img.shape[0] in (1,3,4) or img.shape[-1] in (1,3,4))), \
    f"Plot only 2D gray or RGB(A)-channel-last images. Got shape {img.shape}."

  if ax == None:
    # no whitespace and 1:1 pixel resolution
    fig = plt.figure(frameon=not exact, layout='tight', dpi=5)
    ax = fig.add_axes((0,0,1,1))
    fig.set_size_inches(img.shape[1]/fig.dpi*zoom, img.shape[0]/fig.dpi*zoom)  
    ax.axis('off')

  img = (img - img.min()) / (1e-9+ img.max() - img.min())
  ax.imshow(img, interpolation='none', **imshow_kwargs)
  return ax

# %% # Create Test
def create_test():
  for prop in [0.05, 0.15]:
    ax = image(b.x)

    rm = np.argsort(-p2L)[:int(len(b.l)*prop)]

    np.random.seed(42)
    randoms = np.random.choice(np.delete(np.arange(len(b.l)), rm), int(len(b.l)*prop), replace=False)

    all = list(set(rm) | set(randoms))

    plot.points(ax, b.k, b.l, 5.0/5, lw=1/5)
    plot.points(ax, b.k[all], b.l[all], 5.0/5, lw=2/5, colormap='#00ffff')


    out = {'bad':[], 'random':[], 'i2pi':{}, 'pi2L':p2L.tolist()}
    for i,pi in enumerate(all):
      if pi in rm: out['bad'] += [i]
      else: out['random'] += [i]

      out['i2pi'][i] = int(pi)

      x,y = b.k[pi]
      ax.text(x+12, y, f'{i}', color='cyan', fontsize=150, va='center')

    json.dump(out, open(f'bad-random-{prop}.json', 'w'), indent=2)
    plot.save(ax, f'bad-random-{prop}.png')

# %% # Plot missing points 

missing = [35, 36, 47, 48, 49, 54, 59, 65, 73, 77, 79, 80, 81, 86, 88, 90, 98, 155, 173, 175]

ax = image(b.x)
plot.points(ax, b.k, b.l, 5.0/5, lw=1/5)
plot.points(ax, b.k[missing], b.l[missing], 5.0/5, lw=2/5, colormap='#00ffff')

# add according text labels
for pi in missing:
  x,y = b.k[pi]
  ax.text(x+12, y, f'{pi}', color='cyan', fontsize=150, va='center')

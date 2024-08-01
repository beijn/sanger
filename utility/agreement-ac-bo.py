# %%
import os; os.chdir('..')

# %% # Compare Disagreeing AC and BO annotations on QS_7382

from types import SimpleNamespace as obj

import cellnet.data as data
import cellnet.plot as plot

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from pre_points import load_points_json

I = './data/images/QS_7382.jpg'


cfg = obj(**(dict(
  fraction=1, 
  maxdist=26, 
  rmbad=0,
  sigma=5.0,  # NOTE: do grid search again later when better convergence 
  sparsity=1,
  xnorm_params={},
  xnorm_type='image_per_channel',
  label2int = {'Live Cell':1, 'Dead cell/debris':2, 'Cell':1, 'cell':1},
  ))
)

transforms = A.Compose(transforms=[
    A.Normalize(normalization='image_per_channel'),
    ToTensorV2(transpose_mask=True, always_apply=True)], 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True) 
  )


kp2hm, yunnorm, ymax = data.mk_kp2mh_yunnorm(cfg, norm_using_images=['./data/images/1.jpg'], point_paths=['./data/points.json'])
def mini_batch( who):
    points = load_points_json([f'./data/QS_7382..{who}.json'], [I], cfg.label2int)
    points = {k: v[who] for k,v in points.items()}
    print(type(points), points[I].shape, points[I][0])
    B = next(iter(data.mk_loader([I], 1, transforms, cfg=cfg, override_points=points)))
    return data.batch2cpu(B, z=kp2hm(B))[0]
    
a = mini_batch('ac')
b = mini_batch('bo')

plot.overlay(a.x, a.z, a.m, a.k, a.l)
plot.overlay(b.x, b.z, b.m, b.k, b.l)

# %%

z = a.z*b.z    

ax = plot.image(a.x)
plot.heatmap(a.z, ax, color='#ff0000')
plot.heatmap(b.z, ax, color='#0000ff')
plot.heatmap(z, ax, color='#ff00ff')


# %% 

from scipy.ndimage import label, center_of_mass
l, n = label((z>0.5)*1.0) # type: ignore
co = np.array([center_of_mass(z, l, i)[1::][::-1] for i in range(1, n+1)])

# %% 

ax = plot.image(a.x)
plot.heatmap(a.z, ax, color='#ff0000')
plot.heatmap(b.z, ax, color='#0000ff')
plot.heatmap(z, ax, color='#ff00ff')

plot.points(ax, co, radius=8, colormap='#00ff00', marker='o', lw=5)

# %%
## regenrate masks less points 
def plot_merge_proposal(dl, I):
  ds: CellnetDataset = dl.dataset # type: ignore
  ax = plot.overlay(ds.X[I], None, ds.M[I], None, None, cfg.sigma)
  plot.points(ax, ds.P[I][:,[0,1]], ds.P[I][:,2].astype(int), radius=5, colormap='#ff00ff', marker='o', lw=3)

  ## get all points that are close to an agreeing annotation -> dupes. All others sill be subtracted from the sparsity map
  g = lambda x: np.exp(-x**2 / (2 * cfg.sigma**2))
  cc = np.concatenate
  D = data.Keypoints2Heatmap(cfg.sigma, ynorm=lambda x:x)(a.x,[a.m], agreeing_points[I][:,[0,1]], agreeing_points[I][:,2])
  m = D.max().item()
  D = data.Keypoints2Heatmap(cfg.sigma, ynorm=lambda x:x/m)(a.x,[a.m], agreeing_points[I][:,[0,1]], agreeing_points[I][:,2])
  too_close = {I: np.array([(x,y,l) for (x,y), l in zip(cc([a.k, b.k]), cc([a.l, b.l])) if D[0,int(y),int(x)] < g(5) ])}
  plot.points(ax, too_close[I][:,[0,1]], too_close[I][:,2].astype(int), radius=5, colormap='#00ff00', marker='o', lw=3)
 
  # substract the sparsity map of old_ps from the sparsity map of ds.P (ds.M)
  #... TODO

agreeing_points = {I:np.array([(x,y,1) for x,y in co])}
dl = data.mk_loader([I], 1, transforms, cfg=cfg, override_points=agreeing_points)
plot_merge_proposal(dl, I=I)
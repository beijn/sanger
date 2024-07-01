# %% [markdown]
# # CellNet

# %% # Imports 
IMAGES = 'all'; AUGS = 'val'
#P = 'sigma'; ps = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
P = 'rmbad'; ps = [0.15]
RELEASE = True


import torch
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

import albumentations as A; from albumentations.pytorch import ToTensorV2

import os, ast, json
from types import SimpleNamespace as obj

from util.data import *
import util.plot as plot


CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA else 'cpu'); print('device =', device)
DRAFT = False if os.getenv('BATCHED_RUN', '0')=='1' else True; print('DRAFT =', DRAFT)


key2text = {'tl': 'Training Loss',     'vl': 'Validation Loss', 
            'ta': 'Training Accuracy', 'va': 'Validation Accuracy', 
            'ti': 'Training Image',    'vi': 'Validation Image',
            'lf': 'Loss Function',     'lr': 'Learning Rate',
            'e' : 'Epoch',             'bs': 'Batch Size',
            'fraction': 'Fraction of Data',  'sparsity': 'Artificial Sparsity',  
            'sigma': 'Gaussian Sigma',        'maxdist': 'Max Distance',
            'rmbad': 'Prop. of Difficult Labels Removed'
            }

CROPSIZE=256


cfg_base = obj(
  epochs=(5 if CUDA else 1) if DRAFT else 101 if not RELEASE else 351,
  sigma=5.0,  # NOTE: do grid search again later when better convergence 
  maxdist=26, 
  fraction=1, 
  sparsity=1,
  lr_steps=1.25 if not RELEASE else 2.5,
  lr_gamma=0.1,
  rmbad=0,
)

# %% # Load Data
XNorm, _xmean, _xstd = mk_XNorm([1,2,4])


def mkAugs(mode):
  T = lambda ts:  A.Compose(transforms=[
    A.PadIfNeeded(CROPSIZE, CROPSIZE, border_mode=0, value=0),
    *ts,
    XNorm(), 
    A.PadIfNeeded(CROPSIZE, CROPSIZE, border_mode=0, value=0),
    ToTensorV2(transpose_mask=True, always_apply=True)], 
    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True) 
  )

  vals = [A.D4(),
          ]

  return dict(
    test  = T([]),
    val   = T([A.RandomCrop(CROPSIZE, CROPSIZE, p=1),
               *vals]),
    train = T([A.RandomCrop(CROPSIZE, CROPSIZE, p=1),
               # A.RandomSizedCrop(p=1, min_max_height=(CROPSIZE//2, CROPSIZE*2), height=CROPSIZE, width=CROPSIZE),  # NOTE: issue with resize is that the keypoint sizes will not be updated
               A.Rotate(),
               A.AdvancedBlur(),
               A.Equalize(),
               A.ColorJitter(), 
               A.GaussNoise(),
               *vals])
  )[mode]


# %% # Plot data 
if DRAFT and not CUDA: 
  kp2hm, yunnorm, _ = mk_kp2mh_yunnorm([1,2,4], cfg_base)

  from math import prod
  def plot_grid(grid, **loader_kwargs):
    loader = mk_loader([1], cfg=cfg_base, bs=prod(grid), **loader_kwargs)
    B = next(iter(loader))
    B = batch2cpu(B, z=kp2hm(B))
    for b,ax in zip(B, plot.grid(grid, [CROPSIZE]*2)[1]):
      plot.overlay(b.x, b.z, b.m, b.k, b.l, cfg_base.sigma, ax=ax)

  plot_grid((3,3), transforms=mkAugs('val'))
  plot_grid((3,3), transforms=mkAugs('train'))

  for B in mk_loader([1,2,4], cfg=cfg_base, bs=1, transforms=mkAugs('test'), shuffle=False):
    b = batch2cpu(B, z=kp2hm(B))[0]
    ax = plot.overlay(b.x, b.z, b.m, b.k, b.l, cfg_base.sigma)
 

# %% # Create model 
plt.close('all')

import segmentation_models_pytorch as smp
mk_model = lambda: smp.Unet(  # NOTE TODO: check if spefically used model automatically mirror pads in training or inference
    encoder_name="resnet34" if DRAFT else "resnet152",  # 18 34 50 101 152
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation='sigmoid',
  ).to(device) # type: ignore

# %% # Train 

# def count(y): return yunnorm(y).sum().item()
def accuracy(y,z): 
  ny, nz = y.sum().item(), z.sum().item()
  return 1 - abs(ny - nz) / (nz+1e-9)

def train(epochs, model, optim, lossf, sched, kp2hm, traindl, valdl=None, info={}):
  log = pd.DataFrame(columns='tl vl ta va lr'.split(' '), index=range(epochs))
  def epoch(dl, train):
    l = 0; a = 0; b = 0
    for b, B in enumerate(dl):
      x,m = B['image'].to(device), B['masks'][0].to(device)
      z = kp2hm(B).to(device)

      y = model(x)
      loss = lossf(y*m, z*m) 
      l += loss.item()
      a += accuracy(y*m, z*m)

      if train:
        loss.backward()
        optim.step()
        optim.zero_grad()

    return l/(b+1), a/(b+1)

  for e in range(epochs):
    L = log.loc[e]
    L['lr'] = optim.param_groups[0]['lr']
  
    model.train()
    L['tl'], L['ta'] = epoch(traindl, train=True)
    sched.step() 
  
    if valdl is not None: 
      model.eval()
      with torch.no_grad():
        L['vl'], L['va'] = epoch(valdl, train=False) 

    if DRAFT: plot.train_graph(e, log, info=info, key2text=key2text, clear=True)
  plot.train_graph(epochs, log, info=info, key2text=key2text, accuracy=False) 
  return log

def loss_per_point(b, lossf, kernel=15, exclude=[]):
  loss = lossf.__class__(reduction='none')(*[torch.tensor(x) for x in [b.y, b.z]])
  p2L = np.zeros(len(b.l))
  for i, (l, (x,y)) in enumerate(zip(b.l, b.k)):
    #if l in exclude: continue  # NOTE hack to exclude losses for negative annotations (TODO reevaluate why)
    xx, yy = np.meshgrid(np.arange(loss.shape[2]), np.arange(loss.shape[1]))
    kernel = (xx-x)**2 + (yy-y)**2 < kernel**2
    p2L[i] = (loss * kernel).sum()

  return p2L


splits = [([1], [2])] if DRAFT else\
         [([1,2,4], [1])] if RELEASE else\
         [([1], [2,4])] if IMAGES=='one' else\
         [([2,4], [1]), ([1,4], [2]), ([1,2], [4])] if IMAGES=='all' else\
         []

results = pd.DataFrame()
if not DRAFT: [os.makedirs(_p, exist_ok=True) for _p in ('preds', 'plots')]

def training_run(cfg, traindl, valdl, kp2hm, model=None):
  global results  
  p = cfg.__dict__[P]
  ti,vi = [dl.dataset.ids for dl in (traindl, valdl)]

  if model is None: model = mk_model()
  optim = torch.optim.Adam(model.parameters(), lr=5e-3)
  lossf = torch.nn.MSELoss()
  sched = torch.optim.lr_scheduler.StepLR(optim, step_size=int(cfg.epochs/cfg.lr_steps)+1, gamma=cfg.lr_gamma)

  log = train(cfg.epochs, model, optim, lossf, sched, kp2hm, traindl, valdl, info={P: p})

  _row =  pd.DataFrame(dict(**{P: [p]}, ti=[ti], vi=[vi], **log.iloc[-1]))
  results = _row if results.empty else pd.concat([results, _row], ignore_index=True)
  # TODO CHECK is the saved accuracy correct? Because model ran on sanger reports counts ({'data/4.jpg': 1335.7061956439575, 'data/2.jpg': 1477.7731458703638, 'data/1.jpg': 1791.7817114729842, 'data/3.jpg': 1916.6642587431998}) that are only 0.787571807701338 (0.9790417598444654, 0.7550352604295545, 0.6286384028299937) 


  i2p2L = {}
  # plot and save predictions to disk
  for ii, t in [(ti, 'T'), (vi, 'V')]:
    for i in ii:
      B = next(iter(mk_loader([i], bs=1, transforms=mkAugs('test'), shuffle=False, cfg=cfg)))

      model.eval()
      with torch.no_grad(): y = cpu(model(B['image'].to(device)))
      b = batch2cpu(B, z=kp2hm(B), y=y)[0]
      del B

      if cfg.rmbad != 0: # get the badly predicted points and plot them
        p2L = loss_per_point(b, lossf, kernel=15, exclude=[2])
        if RELEASE or i in vi: 
          i2p2L[i] = p2L  # only save the losses for the validation image 
          print(f'DEBUG: saved point losses for val image {i} (should happen only once per cfg and image)')

      if RELEASE or (vi==[4] and (i in (1,4))):  # plot T1 and V4 for all [1,2]|[4] runs
        ax1 = plot.overlay(b.x, b.y, b.m, b.k, b.l, cfg.sigma) 
        ax2 = plot.diff   (b.y, b.z, b.m, b.k, b.l, cfg.sigma)

        if cfg.rmbad != 0: 
          rm = np.argsort(-p2L[i])[:int(len(b.l)*cfg.rmbad)]
          [plot.points(a, b.k[rm], b.l[rm], color='#00ff00', lw=3)
            for a in (ax1, ax2)]

        if not DRAFT: 
          id = f"{P}={p}-{t}{i}"
          #np.save(f'preds/{id}.npy', y)
          plot.save(ax1, f'plots/{id}.pred.png')
          plot.save(ax2, f'plots/{id}.diff.png')
          plt.close('all') # save but don't show

  print(i2p2L)
  return dict(model=model, log=log, i2p2L=i2p2L)


loader = lambda c, ids, mode: mk_loader(ids, bs=1 if mode=='test' else 16, shuffle=False, cfg=c,
    transforms=mkAugs(('val' if AUGS=='train' else 'test') if mode=='_val_' else mode)) 
kp2hm, yunnorm, _ymax = mk_kp2mh_yunnorm([1,2,4], cfg_base)

for p in [ps[-1]] if DRAFT else ps:
  cfg = obj(**(cfg_base.__dict__ | {P: p}))
  if P in ['sigma']: kp2hm, yunnorm, _ymax = mk_kp2mh_yunnorm([1,2,4], cfg)

  i2p2L = {}

  for ti, vi in splits:
    cfg = obj(**(cfg.__dict__ | dict(ti=ti, vi=vi)))

    traindl, valdl = loader(cfg, ti, AUGS), loader(cfg, vi, '_val_')

    out = training_run(cfg, traindl, valdl, kp2hm)
    i2p2L |= out['i2p2L'] # NOTE: overrides if image in multiple val sets

  if cfg.rmbad != 0:  # remove the bad points and retrain
    keep = {i: np.argsort(-p2L)[int(len(p2L)*cfg.rmbad):] for i,p2L in i2p2L.items()} 
    
    for _i, k in keep.items(): print(f"DEBUG: keeping {len(k)} of {len(i2p2L[_i])} points for {_i}")

    for ti, vi in splits:
      cfg = obj(**(cfg.__dict__ | dict(ti=ti, vi=vi, epochs=cfg.epochs//2+1, rmbad=0.1)))

      traindl, valdl = loader(cfg, ti, AUGS), loader(cfg, vi, '_val_')
      for dl in [traindl, valdl]: 
        ds: CellnetDataset = dl.dataset # type: ignore
        ds.P = {i: ds.P[i][keep[i]] for i in ds.P}
        ds.L = {i: ds.L[i][keep[i]] for i in ds.L}
        ds._generate_masks(fraction=1, sparsity=1)
        # regenerate masks, but don't throw away more data (f,s=1)
        # NOTE: because we do it for each split repeatedly its a waste of compute. More efficient: to do it once but would need a compley refactor
      
      out = training_run(cfg, traindl, valdl, kp2hm, model=out['model'])
  
# %%
if RELEASE: # save model to disk
  B = next(iter(mk_loader([1], cfg=cfg_base, bs=1, transforms=mkAugs('test'), shuffle=False)))
  x = batch2cpu(B)[0].x[None]
  
  m = out['model'] # type: torch.nn.Module # type: ignore 
  m.eval()

  # save a test in/out
  np.save('.cache/export_test_x_1.npy', x)
  np.save('.cache/export_test_y_1.npy', cpu(m(gpu(x, device=device))))

  m.save_pretrained('./model_export')  # specific to master branch of SMP. TODO: make more robust with onnx. But see problem notes in cellnet.yml
  os.remove('model_export/README.md')

  with open('model_export/pipeline.json', 'w') as f: json.dump(dict(xmean=_xmean, xstd=_xstd, ymax=_ymax), f, indent=2)


# %% # save the results as csv. exclude model column; plot accuracies

if not DRAFT:
  results.to_csv('results.csv', index=False, sep=';')
  R = pd.read_csv('results.csv', sep=';', converters=dict(ti=ast.literal_eval, vi=ast.literal_eval)).rename(columns=dict(vi=key2text['vi']))
  plot.regplot(R, P, key2text)

results # type: ignore
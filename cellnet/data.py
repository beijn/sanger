# NOTE on data format: this is all BHWC. Except Keypoints2Heatmap which is HWC. Torch needs BCHW, albumentations makes the conversions


import pathlib
import numpy as np, cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt

import torch, torch.utils.data

import json, os
import albumentations as A
from types import SimpleNamespace as obj


noneor = lambda x, d: x if x is not None else d 

mapd = lambda d,f: {k:f(v) for k,v in d.items()}
dict2stack = lambda D: np.stack([D[i] for i in sorted(D.keys())], axis=0)
stack2dict = lambda S, ids: {i:S[idx] for idx,i in enumerate(ids)}
wrapDictAsStack = lambda f, ids: lambda D: stack2dict(f(dict2stack(D)), ids)


def gpu(x, device): return torch.from_numpy(x).float().to(device)
def cpu(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x) if isinstance(x, list) else x


def imgid(path): return pathlib.Path(path).stem

def batched(f):
  def inner(B):
    n_masks = len(B['masks'])
    R= torch.stack([f(
          image=b[0], masks=[b[m] for m in range(1,n_masks+1)], keypoints=b[n_masks+1], class_labels=b[n_masks+2]
        ) for b in zip(B['image'], *B['masks'], B['keypoints'], B['class_labels'])
      ], dim=0)
    return R
  return inner

batch2cpu = lambda B, z=None, y=None: [obj(**{k:cpu(v) for v,k in zip(b, 'xmklzy')}) 
              for b in zip(B['image'], B['masks'][0], B['keypoints'], B['class_labels'], 
              *([] if z is None else [z]), *([] if y is None else [y]))]


def load_images(paths): return {imgid(p): cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths}

def load_points(path, ids, l2i):
  P = {i:[] for i in ids}; L = {i:[] for i in ids}
  for entry in (raw := json.load(open(path))):
    image_id = int(entry['data']['img'].split('/')[-1][9:].split('.')[0])
    if image_id not in ids: continue
    for annotation in entry['annotations']:
      for result in annotation['result']:
        x = result['value']['x']/100 * result['original_width']
        y = result['value']['y']/100 * result['original_height']
        l = result['value']['keypointlabels'][0]
        P[image_id] += [(x,y)]
        L[image_id] += [l2i(l)]
  return [mapd(v, np.array) for v in (P,L)]


class CellnetDataset(torch.utils.data.Dataset):
  def __init__(self, image_paths, sigma, maxdist, sparsity=1.0, fraction=1.0, batch_size=None, transforms=None, 
               point_annotations_file='data/points.json', label2int=lambda l:{'Live Cell':1, 'Dead cell/debris':2}[l], 
               **_junk):
    super().__init__()
    self.batch_size = noneor(batch_size, len(image_paths))
    self.maxdist=maxdist; self.fraction=fraction; self.sparsity=sparsity
    self.ids=[imgid(p) for p in image_paths]; self.sigma=sigma; self.label2int=label2int; self.transforms = transforms if transforms else lambda **x:x
    self.X = load_images(image_paths)  # NOTE albumentations=BHWC 可是 torch=BCHW
    self.P, self.L = load_points(point_annotations_file, self.ids, label2int)
    
    self._generate_masks(fraction=self.fraction, sparsity=self.sparsity)

  def _generate_masks(self, fraction=1.0, sparsity=1.0):
    assert fraction>0 and sparsity>0, "fraction and sparsity must be positive (0,1]"

    self.M = mask_sparse(self.ids, self.X, self.P, self.L, self.maxdist)

    if (f:=fraction) < 1.0: 
      _s = self.X[self.ids[0]].shape
      x,y = int(_s[0]*f), int(_s[1]*f)
      self.X = mapd(self.X, lambda a: a[:x,:y])
      self.M = mapd(self.M, lambda a: a[:x,:y])

    # filter out all points outside of X
    for i in self.ids:
      P = []; L = []
      for ((x,y), l) in zip(self.P[i], self.L[i]):
        if  0 <= x < self.X[i].shape[1]\
        and 0 <= y < self.X[i].shape[0]:
          P += [(x,y)]; L += [l]
      self.P[i] = np.array(P); self.L[i] = np.array(L)

    if (s:=sparsity) < 1.0:
      self.P = mapd(self.P, lambda a: a[::int(1/s)])
      self.L = mapd(self.L, lambda a: a[::int(1/s)])
   
  def get(self, n): return getattr(self, n)
  def set(self, n, to): setattr(self, n, to)

  def __len__    (self): return max(self.batch_size, len(self.X))
  def __getitem__(self, i): 
    i = self.ids[i % len(self.X)]
    return self.transforms(image=self.X[i], masks=[self.M[i]], keypoints=self.P[i], class_labels=self.L[i])

  def map(self, f, dim='XY'):
    for n in dim: self.set(n, f(self.get(n)))


def mk_loader(image_paths, bs, transforms, cfg, shuffle=True):
  def collate(S):
    return dict(
      image = torch.stack([s['image'] for s in S]),
      masks = [torch.stack([s['masks'][i] for s in S]) for i in range(len(S[0]['masks']))],
      keypoints = [s['keypoints'] for s in S],
      class_labels = [s['class_labels'] for s in S],
    )

  if image_paths=='all': image_paths = cfg.annotated_images
  from torch.cuda import device_count as gpu_count; from multiprocessing import cpu_count 
  return torch.utils.data.DataLoader(CellnetDataset(image_paths, transforms=transforms, batch_size=bs, **cfg.__dict__), 
    batch_size=bs, shuffle=shuffle, collate_fn=collate, pin_memory=True, num_workers=8) # TODO: figure out and fix error and change back #8 if torch.cuda.is_available() else 2)


def mk_XNorm(cfg, norm_using_images='all'):
  if norm_using_images == 'all': norm_using_images = cfg.annotated_images
  X = dict2stack(load_images(norm_using_images))
  params = dict(
    mean = list(X.mean(axis=(0,1,2))/255),
    std  = list(X.std (axis=(0,1,2))/255),
  )

  return (lambda **kw: A.Normalize(normalization='standard' if cfg.xnorm_type=='imagenet' else cfg.xnorm_type, **params, **kw)), params


def mk_kp2mh_yunnorm(cfg, norm_using_images='all'):
  """Z-score norm improves DNN training according to @lecun2002efficient. BWHC"""
  if norm_using_images == 'all': norm_using_images = cfg.annotated_images
  ds = CellnetDataset(norm_using_images, **cfg.__dict__)

  Y = np.stack([Keypoints2Heatmap(cfg.sigma, ynorm=lambda y:y, labels_to_include=[1])(x.transpose(2,0,1),[m],p,l) 
                for x,m,p,l in zip(*[v.values() for v in [ds.X, ds.M, ds.P, ds.L]])], axis=0)
  
  ymax = Y.max() 
  ynorm   = lambda y: y/ymax  # norm to (0,1]
  yunnorm = lambda y: y*ymax
  # Y = ((Y - ymean) / ystd).astype(np.float32)  # unit norm, using dataset wide mean and std

  kp2mh = batched(Keypoints2Heatmap(cfg.sigma, ynorm, labels_to_include=[1]))
  return kp2mh, yunnorm, ymax


# HWC - this function works not with a batch dimension
def Keypoints2Heatmap(sigma, ynorm, labels_to_include=[1]):
  def f(image, masks, keypoints, class_labels):
    hw = image.shape[1:3] 
    if len(keypoints)==0: return torch.zeros(1,*hw).float()  # no keypoints
    Y = onehot(hw, [[(x,y) for x,y,*_ in keypoints]], [class_labels])
    Y = Y[0][:,:,[l-1 for l in labels_to_include]]  # the [B][H,W,[Cs]] form is very important
    for c in range(Y.shape[-1]):
      Y[...,c] = gaussian_filter(Y[...,c], sigma=sigma, mode='constant', cval=0)
    return torch.from_numpy(ynorm(Y)).permute(2,0,1).to(torch.float32)  # HWC -> CHW
  return f


# this function expects a batch dim in P and L
def onehot(hw, P, L):
  cs = list(set.union({0}, *[set(ls) for ls in L]))
  A = np.zeros((len(P),*hw, max(cs))) 
  for b in range(len(P)):
    for (x,y), l in zip(P[b], L[b]):
      A[b, int(y), int(x), l-1] = 1
  return A  # -> BHWC

def mask_sparse(ids, X, P, L, maxdist): 
  D = onehot(X[ids[0]].shape[0:2], list(P.values()), list(L.values()))
  D = D.sum(axis=-1)  # all types of points are treated the same => we dont include the points for negative examples but we train on their image parts! :]
  for b in range(len(P)):
    D[b] = distance_transform_edt(1-D[b])
  D = (D > maxdist).reshape(D.shape)
  B = dict2stack(mk_fgmask(ids, X, thresh=0.01))
  return stack2dict(1-(B & D)[:,:,:,None].astype(np.float32), ids)

def mk_fgmask(ids, X, thresh=0.01):
  def create_masks(X):
    """X: BHWC"""
    def get_sess(model_path: str):
      import onnxruntime as ort
      providers = [
        ( "CUDAExecutionProvider",
          { "device_id": 0,
            "gpu_mem_limit": int(8000 * 1024 * 1024),  # in bytes
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        },),
        "CPUExecutionProvider" ]
      opts: ort.SessionOptions = ort.SessionOptions()
      opts.log_severity_level = 2; opts.log_verbosity_level = 2
      opts.inter_op_num_threads = 1;  opts.intra_op_num_threads = 1  # otherwise will fail on slurm
      return ort.InferenceSession(model_path, providers=providers, sess_options=opts)
    model_path = "data/phaseimaging-combo-v3.onnx"
    X = np.transpose(X,(0,3,1,2))[:,[2,1],:,:].astype(np.float32)  # now BCHW, and only B and G channels
    X = (X - X.min()) / (X.max() - X.min())  # NOTE this norm is very relevant for the model to work TODO make it more robust 
    pred = (sess := get_sess(model_path=model_path)).run(
      output_names = [out.name for out in sess.get_outputs()], 
      input_feed   = {sess.get_inputs()[0].name: X})[0]
    return (pred > thresh).reshape(len(X), *X.shape[2:]).astype(np.uint8)
  
  # if data/masks/fgmasks.npy exists, load it, otherwise create it
  cachedir = os.path.expanduser('~/.cache/cellnet')
  try: return {i: np.load(f'{cachedir}/fgmasks/{i}.npy') for i in ids}
  except FileNotFoundError: 
    M = wrapDictAsStack(create_masks, ids)(X)
    os.makedirs(f'{cachedir}/fgmasks', exist_ok=True)
    for i in ids: np.save(f'{cachedir}/fgmasks/{i}.npy', M[i])
    return M

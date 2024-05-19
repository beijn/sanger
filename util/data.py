# NOTE on data format: this is all BHWC. Except Keypoints2Heatmap which is HWC. Torch needs BCHW, albumentations makes the conversions


import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, distance_transform_edt

from sympy import O
import torch
import torch.utils.data

from collections import defaultdict 
from types import SimpleNamespace as obj
import json
import albumentations as A


noneor = lambda x, d: x if x is not None else d 

class CellnetDataset(torch.utils.data.Dataset):
  def __init__(self, image_ids, sigma, sparsity=1.0, fraction=1.0, batch_size=None, maxdist=1e9, transforms=None, point_annotations_file='data/points.json', 
               label2int=lambda l:{'Live Cell':1, 'Dead cell/debris':2}[l], **_junk):
    super().__init__()
    self.batch_size = noneor(batch_size, len(image_ids))
    self.image_ids=image_ids; self.sigma=sigma; self.label2int=label2int; self.transforms = transforms if transforms else lambda **x:x
    self.X = load_images(image_ids)  # NOTE albumentations=BHWC 可是 torch=BCHW
    self.P, self.L = load_points(point_annotations_file, image_ids, label2int)
    self.M = mask_sparse(self.X, self.P, self.L, maxdist)

    if (f:=fraction) < 1.0: 
      x,y = int(self.X.shape[1]*f), int(self.X.shape[2]*f)
      self.X = self.X[:,:x,:y,:]
      self.M = self.M[:,:x,:y,:]

    # filter out all points outside of X
    for i in range(len(self.X)):
      P = []; L = []
      for ((x,y), l) in zip(self.P[i], self.L[i]):
        if  0 <= x < self.X[i].shape[1]\
        and 0 <= y < self.X[i].shape[0]:
          P += [(x,y)]; L += [l]
      self.P[i] = np.array(P); self.L[i] = np.array(L)

    if (s:=sparsity) < 1.0:
      self.P = [p[::int(1/s)] for p in self.P]
      self.L = [l[::int(1/s)] for l in self.L]
   
  def get(self, n): return getattr(self, n)
  def set(self, n, to): setattr(self, n, to)

  def __len__    (self): return max(self.batch_size, len(self.X))
  def __getitem__(self, i): 
    i = i % len(self.X)
    return self.transforms(image=self.X[i], masks=[self.M[i]], keypoints=self.P[i], class_labels=self.L[i])

  def map(self, f, dim='XY'):
    for n in dim: self.set(n, f(self.get(n)))

# HWC - this function works not with a batch dimension
def Keypoints2Heatmap(sigma, ymean, ystd, labels_to_include=[1]):
  def f(image, masks, keypoints, class_labels):
    hw = image.shape[1:3] 
    if len(keypoints)==0: return torch.zeros(1,*hw).float()  # no keypoints
    Y = onehot(hw, [[(x,y) for x,y,*_ in keypoints]], [class_labels])
    Y = Y[0][:,:,[l-1 for l in labels_to_include]]  # the [B][H,W,[Cs]] form is very important
    for c in range(Y.shape[-1]):
      Y[...,c] = gaussian_filter(Y[...,c], sigma=sigma, mode='constant', cval=0)
    Y = ((Y - ymean) / ystd).astype(np.float32)
    # norm to 0-1
    Y -= Y.min()
    Y /= Y.max()
    return torch.from_numpy(Y).permute(2,0,1)  # HWC -> CHW
  return f


def load_images(ids): return np.stack([cv2.cvtColor(cv2.imread(f'data/{i}.jpg'), cv2.COLOR_BGR2RGB) for i in ids], axis=0)

def load_points(path, ids, l2i):
  P = [[] for _ in ids]; L = [[] for _ in ids]
  for entry in (raw := json.load(open(path))):
    image_id = int(entry['data']['img'].split('/')[-1][9:].split('.')[0])
    if image_id not in ids: continue
    i = ids.index(image_id)
    for annotation in entry['annotations']:
      for result in annotation['result']:
        x = result['value']['x']/100 * result['original_width']
        y = result['value']['y']/100 * result['original_height']
        l = result['value']['keypointlabels'][0]
        P[i] += [(x,y)]
        L[i] += [l2i(l)]
  return [np.array(ps) for ps in P], [np.array(ls) for ls in L]

def mk_norm(x):
  """Z-score norm improves DNN training according to @lecun2002efficient. BWHC"""
  m = x.mean(axis=(0,1,2), keepdims=True)
  s = x.std(axis=(0,1,2), keepdims=True)
  return 

# this function expects a batch dim in P and L
def onehot(hw, P, L):
  cs = list(set.union({0}, *[set(ls) for ls in L]))
  A = np.zeros((len(P),*hw, max(cs))) 
  for b in range(len(P)):
    for (x,y), l in zip(P[b], L[b]):
      A[b, int(y), int(x), l-1] = 1
  return A  # -> BHWC

def mask_sparse(X, P, L, maxdist): 
  D = onehot(X.shape[1:3], P, L)
  D = D.sum(axis=-1)  # all types of points are treated the same => we dont include the points for negative examples but we train on their image parts! :]
  for b in range(len(P)):
    D[b] = distance_transform_edt(1-D[b])
  D = (D > maxdist).reshape(D.shape)
  B = mk_fgmask(X, thresh=0.01)
  return 1-(B & D)[:,:,:,None].astype(np.float32)

def mk_fgmask(X, thresh=0.01):
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




''' # DECRAP 
    #padding = lambda x: np.pad(x, [(0,0), (pad,pad), (pad,pad), (0,0)], 'reflect') 
    #for n in 'XM': self.set(n, padding(self.get(n)))

def load_point_annotations(path):  # DECRAP
  data = defaultdict(lambda: defaultdict(list)) #data: image_id -> label -> array of instance x X x Y
  for entry in (raw := json.load(open(path))):
    image_id = int(entry['data']['img'].split('/')[-1][9:].split('.')[0])
    for annotation in entry['annotations']:
      for result in annotation['result']:
        x = result['value']['x']/100 * result['original_width']
        y = result['value']['y']/100 * result['original_height']
        label = result['value']['keypointlabels'][0]
        if x < result['original_width'] and y < result['original_height']:
          data[image_id][label].append(np.array([x,y]))
  for image_id in data:
    for label in data[image_id]:
      data[image_id][label] = np.stack(data[image_id][label])
  return data

def annot2onehot(hw, image_ids, points):
  A = np.zeros((len(image_ids),*hw))
  for i, id in enumerate(image_ids):
    for label in points[id]:
      for point in points[id][label]:
        A[i, int(point[1]), int(point[0])] = 1 if label == 'Live Cell' else 0
  return A


def mk_heatmaps(hw, image_ids, points, sigma):
  Y = annot2onehot(hw, image_ids, points)[:,:,:,None]
  for b in range(len(image_ids)):
    Y[b] = gaussian_filter(Y[b], sigma=sigma, mode='constant', cval=0)  
  return Y

  


def mk_mask_not_annotated(X, P, maxdist): 
  D = annot2onehot(X.shape[1:3], P)
  for b in range(len(image_ids)): 
    D[b] = distance_transform_edt(1-D[b])
  D = (D > maxdist).reshape(D.shape)
  B = mk_fgmask(X, thresh=0.01)
  return (B & D)[:,:,:,None]   


def mk_fgmask(X, thresh=0.01):
  """X: BHWC"""
  
  def doitwithonnx(X):
    import onnxruntime as ort
    import numpy as np

    def get_sess(model_path: str):
      providers = [
        ( "CUDAExecutionProvider",
          { "device_id": 0,
            "gpu_mem_limit": int(8000 * 1024 * 1024),  # in bytes
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        },),
        "CPUExecutionProvider" ]
      sess_opts: ort.SessionOptions = ort.SessionOptions()
      sess_opts.log_severity_level = 2; sess_opts.log_verbosity_level = 2

      # without these it fails on HPC
      # from multiprocessing import cpu_count 
      # sess_opts.inter_op_num_threads = cpu_count()//6
      # sess_opts.intra_op_num_threads = cpu_count()//6  # help - ask Sten
      
      sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)
      return sess

    model_path = "data/phaseimaging-combo-v3.onnx"

    X = np.transpose(X,(0,3,1,2))[:,[2,1],:,:].astype(np.float32)  # now BCHW
    X = (X - X.min()) / (X.max() - X.min()) 

    pred = (sess := get_sess(model_path=model_path)).run(
      output_names = [out.name for out in sess.get_outputs()], 
      input_feed   = {sess.get_inputs()[0].name: X})[0]
    del sess

    mask = (pred > thresh).reshape(len(X), *X.shape[2:])
    return mask

  from multiprocessing import cpu_count 
  return np.zeros_like(X[:,:,:,0]) # if cpu_count() > 8 else doitwithonnx(X)  # TODO: fix onnx for HPC
'''
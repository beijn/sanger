
# %%
import torch
import segmentation_models_pytorch as smp

import cv2
import numpy as np

import json 


TEST = True
PLOT = False  # NOTE uses the cellnet environment not cellnet_release
if PLOT:
  import util.plot as plot


# %% 

model_settings = json.load(open('model_export/pipeline.json'))


def load_images(ids, mean, std, max_pixel_value=255.0): 
  mean, std = [np.array(x, dtype=np.float32) for x in (mean, std)]
  return {i: ((cv2.cvtColor(cv2.imread(f'data/{i}.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32)  # NOTE: these castings are weird but important. can probably do them more reasonably 
              - mean * max_pixel_value) / (std * max_pixel_value)).transpose(2, 0, 1) for i in ids}

X = load_images([1], mean=model_settings['xmean'], std=model_settings['xstd'])[1][None]

if PLOT:
  pass#plot.image(X[0])

# %% 

model = smp.Unet.from_pretrained('./model_export')
model.eval()  # important

# %% 

Y = model(torch.tensor(X).float()).detach().cpu().numpy()

if PLOT: 
  plot.overlay(X[0], Y[0])


# %% 
if TEST:
  _X = np.load('.cache/export_test_x_1.npy')
  _Y = np.load('.cache/export_test_y_1.npy')

  if PLOT: 
    plot.diff(X[0], _X[0])
    plot.diff(Y[0], _Y[0])

  # NOTE: due to weird casting differences which I didn't figure out yet _X and X are a little different
  print('deviation between train and relase in X', abs((_X - X).sum()))
  print('deviation between train and relase in Y', abs((_Y - Y).sum()))
  assert abs((_X - X).sum()) < 1e-2
  assert abs((_Y - Y).sum()) < 1e-2
    

  

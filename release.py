import torch
import segmentation_models_pytorch as smp

import cv2
import numpy as np

import json

def load_image(path, pipline_settings): 
  mean, std = [np.array(pipline_settings[k], dtype=np.float32)  * 255   for k in ('xmean', 'xstd')]
  return ((cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) - mean) / std).transpose(2, 0, 1)
  # NOTE: these casts are weird but important. can probably do them more reasonably 

def count(image_paths):
  """More efficient if run with all image paths at once."""
  pipline_settings = json.load(open('model_export/pipeline.json'))

  model = smp.Unet.from_pretrained('./model_export')
  model.eval()  # important

  counts = {}
  for path in image_paths:
    X = load_image(path, pipline_settings)[None]
    Y = model(torch.tensor(X).float()).detach().cpu().numpy()

    counts[path] = np.sum(Y)*pipline_settings["ymax"]
  return counts


if __name__ == '__main__':
  import sys
  import os
  folder = sys.argv[1]
  image_paths = [f"{folder}/{f}" for f in os.listdir(folder) if f.endswith('.jpg')]

  print(count(image_paths))

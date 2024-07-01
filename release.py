from pyexpat import model
import torch
import segmentation_models_pytorch as smp

import cv2
import numpy as np

import json
import requests, zipfile, tqdm, shutil, os


def get_latest_release(repo='beijn/cellnet'):
  return requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()["tag_name"]

def download(url, filename, overwrite=False):
  if os.path.isdir(filename): filename = os.path.join(filename, url.split('/')[-1])
  if not overwrite and os.path.isfile(filename): return print("Loading cached model.")
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  try: 
    with open(filename, 'wb') as f:   
      with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))

        with tqdm.tqdm(desc='Caching model', total=total, unit='B', unit_scale=True, unit_divisor=1024) as pb:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pb.update(len(chunk))
  except Exception as e:
    os.remove(filename)
    raise e
  

# TODO: restrict to compatible versions?
def init_model(version:str='latest', keep_download_cache=True):
  cache = os.path.expanduser('~/.cache/cellnet')
  modeldir = f'{cache}/model_export'
  versionfile = f'{modeldir}/version.json'
  if version == 'latest': version = get_latest_release()

  prexisting_version = json.load(open(versionfile))['version'] if os.path.isfile(versionfile) else None
  if prexisting_version != version:
    # TODO checksum of model.zip
    download(f'https://github.com/beijn/cellnet/releases/download/{version}/model.zip', f'{cache}/model-{version}.zip', overwrite=True)   
    if os.path.isdir(modeldir): shutil.rmtree(modeldir)
    with zipfile.ZipFile(f'{cache}/model-{version}.zip', 'r') as zip_ref:
      zip_ref.extractall(cache)
    if not keep_download_cache: os.remove(f'{cache}/model-{version}.zip')

    with open(versionfile, 'w') as f: json.dump({'version': version}, f)

  pipeline_settings = json.load(open(f'{modeldir}/pipeline.json'))
  model = smp.Unet.from_pretrained(f'{modeldir}')
  setattr(model, 'pipline_settings', pipeline_settings)
  return model

def load_image(path, pipline_settings): 
  mean, std = [np.array(pipline_settings[k], dtype=np.float32)  * 255   for k in ('xmean', 'xstd')]
  return ((cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) - mean) / std).transpose(2, 0, 1)[:,:256,:256]
  # NOTE: these casts are weird but important. can probably do them more reasonably 

def count(image_paths, model=None):
  if model is None or type(model) == str:  
    model = init_model('latest' if model is None else model)
  model.eval()  # important

  counts = {}
  for path in image_paths:
    X = load_image(path, model.pipline_settings)[None]
    Y = model(torch.tensor(X).float()).detach().cpu().numpy()

    counts[path] = np.sum(Y)*model.pipline_settings["ymax"]
  return counts


if __name__ == '__main__':
  import sys
  import os
  folder = sys.argv[1]
  image_paths = [f"{folder}/{f}" for f in os.listdir(folder) if f.endswith('.jpg')]

  model = init_model('v0.0.2')
  counts = count(image_paths, model)
  print(counts)
  json.dump(counts, open('counts.json', 'w'), indent=2, sort_keys=True)

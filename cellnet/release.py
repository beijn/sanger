import torch
import segmentation_models_pytorch as smp

from PIL import Image
import numpy as np

import json, zipfile, shutil, os, sys

import cellnet
import cellnet.plot as plotting
from cellnet.internet import download, GHAPI


GH = GHAPI('beijn/cellnet')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
  
def get_latest_compatible_model_version():
  try: versions = GH.get_all_releases()
  except Exception as e: sys.exit(print(f'Could not get the latest release from GH API. Please specify manually in init_model(<version>). See available versions at https://github.com/beijn/cellnet/releases.\n\n{e}'))

  compatible_versions = [v for v in versions if v.split('-')[0]==cellnet.__model_api_version__]
  if not compatible_versions: sys.exit(print(f"No compatible model (model api version {cellnet.__model_api_version__}) found at https://github.com/beijn/cellnet/releases. \nPlease specify manually in init_model(<version>). \nFound incompatible versions: {versions}"))

  return compatible_versions[0]


# TODO: restrict to compatible versions?
def init_model(version:str|None='latest', keep_download_cache=True):
  cache = os.path.expanduser('~/.cache/cellnet')
  modeldir = version if type(version) is str and os.path.isdir(version) else f'{cache}/model_export'
  versionfile = f'{modeldir}/version.json'

  # if the version is None, we just use whatever is cached or redownload the latest if it's not cached
  if not (version == None and os.path.isdir(modeldir)): 
    if version == None: version = 'latest'
    if version == 'latest': version = get_latest_compatible_model_version()

    prexisting_version = json.load(open(versionfile))['version'] if os.path.isfile(versionfile) else None
    if prexisting_version != version:
      # TODO checksum of model.zip
      download(f'https://github.com/beijn/cellnet/releases/download/{version}/model.zip', f'{cache}/model-{version}.zip', overwrite=True)   
      if os.path.isdir(modeldir): shutil.rmtree(modeldir)
      with zipfile.ZipFile(f'{cache}/model-{version}.zip', 'r') as zip_ref:
        zip_ref.extractall(cache)
      if not keep_download_cache: os.remove(f'{cache}/model-{version}.zip')

      with open(versionfile, 'w') as f: json.dump({'version': version}, f)

  settings = json.load(open(f'{modeldir}/settings.json'))
  model = smp.Unet.from_pretrained(f'{modeldir}')
  setattr(model, 'settings', settings)
  setattr(model, 'version', version)
  print('Model version:', version)
  return model.to(DEVICE)

def load_image(image_file_descriptor, model_settings):
  X = np.array(Image.open(image_file_descriptor))
  match model_settings['xnorm_type']:
    case 'imagenet': 
      m, s = [np.array(model_settings['xnorm_params'][k], dtype=np.float32)  * 255   for k in ('mean', 'std')]
      return ((X - m) / s).transpose(2, 0, 1)
    case 'image_per_channel':  
      m, s = X.mean(axis=(0,1)), X.std(axis=(0,1))
      return ((X - m) / s).transpose(2, 0, 1)
    case other: raise ValueError(f"Unknown xnorm_type '{other}' in model.settings")


def count(images, model=None, plot=True):
  if model is None or type(model) == str:  
    model = init_model(model)
  model.eval()  # important

  counts = {}; plots = {}
  for image in images:
    X = load_image(image, model.settings)[None]
    Y = model(torch.tensor(X).float().to(DEVICE)).detach().cpu().numpy()

    counts[image.name] = np.sum(Y)*model.settings["ymax"]
    if plot: plots[image.name] = plotting.overlay(X[0], Y[0])

  return counts, plots


if __name__ == '__main__':
  target = sys.argv[1]
  model_version = sys.argv[2] if len(sys.argv) > 2 else None

  # if target is a folder 
  if os.path.isdir(target):
    image_paths = [f"{target}/{f}" for f in os.listdir(target) if f.endswith('.jpg')]
  elif os.path.isfile(target):
    image_paths = [target]
  else: raise ValueError(f"Could not find target: {target}") 

  model = init_model(model_version)

  counts = {}
  os.makedirs('plots', exist_ok=True)
  for path in image_paths: 
    with open(path, 'rb') as f:
      cs, plots = count([f], model, plot=True)
      counts |= cs 
      plotting.save(plots[path], os.path.join('plots', os.path.basename(path.removesuffix('.jpg')+'.png')))

  print(counts)

  if os.path.isfile('counts.json'): 
    counts = json.load(open('counts.json')) | counts
  json.dump(counts, open('counts.json', 'w'), indent=2, sort_keys=True)

import re
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


def is_compatible(model_version: str):
  return model_version.split('-')[0] == cellnet.__model_api_version__
                            
  
def get_newest_compatible_model_version():
  try: versions = GH.get_all_releases()
  except Exception as e: sys.exit(print(f'Could not get the latest release from GH API. Please specify manually in init_model(<version>). See available versions at https://github.com/beijn/cellnet/releases.\n\n{e}'))

  compatible_versions = [v for v in versions if is_compatible(v)]
  if not compatible_versions: sys.exit(print(f"No compatible model (model api version {cellnet.__model_api_version__}) found at https://github.com/beijn/cellnet/releases. \nPlease specify manually in init_model(<version>). \nFound incompatible versions: {versions}"))

  return compatible_versions[0]


# TODO: restrict to compatible versions?
def init_model(version:str|None='latest', keep_download_cache=True):
  cache = './.cache/cellnet'
  os.makedirs(cache, exist_ok=True)
  modeldir = version if type(version) is str and os.path.isdir(version) else f'{cache}/model_export'
  versionfile = f'{modeldir}/version.json'
  prexisting_version = json.load(open(versionfile))['version'] if os.path.isfile(versionfile) else None
  if version == None: version = 'latest'
  islatest = version == 'latest'

  # if the version is None, we just use whatever is cached or redownload the latest if it's not cached
  if not (version == None and os.path.isdir(modeldir)): 
    if version == 'latest': 
      version = GH.get_latest_release()
      if not is_compatible(version):
        version = get_newest_compatible_model_version()
    elif not is_compatible(version): sys.exit(print(f"ERROR: Model version {version} is not compatible with the current model API version {cellnet.__model_api_version__}. Please specify a compatible model version (eg. 'latest') or use a matching software version."))

    if prexisting_version != version:
      cached_versions = [re.match(r'model-(.*)\.zip', d) for d in os.listdir(cache)]      
      cached_versions = [m.group(1) for m in cached_versions if m]
      
      if version not in cached_versions:
        # TODO checksum of model.zip
        releases_url = 'https://github.com/beijn/cellnet/releases'
        try: download(f'{releases_url}/download/{version}/model.zip', f'{cache}/model-{version}.zip', f'Model', overwrite=True)  
        except FileNotFoundError as e: sys.exit(print(f'Could not download model version `{version}`. Please check that it exists at {releases_url}.'))

      if os.path.isdir(modeldir): shutil.rmtree(modeldir)
      with zipfile.ZipFile(f'{cache}/model-{version}.zip', 'r') as zip_ref:
        zip_ref.extractall(cache)
      if not keep_download_cache: os.remove(f'{cache}/model-{version}.zip')

      with open(versionfile, 'w') as f: json.dump({'version': version}, f)
  else: version = prexisting_version

  settings = json.load(open(f'{modeldir}/settings.json'))

  model_class = settings['model_architecture'].split(':')[0]
  model = {'smp.Unet': smp.Unet, 
           'smp.UnetPlusPlus': smp.UnetPlusPlus,
           'smp.FPN': smp.FPN,
           'smp.PSPNet': smp.PSPNet,
           'smp.DeepLabV3': smp.DeepLabV3,
           'smp.DeepLabV3Plus': smp.DeepLabV3Plus,
           'smp.Linknet': smp.Linknet,
           'smp.MAnet': smp.MAnet,
           'smp.PAN': smp.PAN,
           }[model_class].from_pretrained(f'{modeldir}')  ## TODO make class dynamic depending on settings
  
  
  setattr(model, 'settings', settings)
  setattr(model, 'version', version)
  print('Model version:', version, '(latest)' if islatest else '(cached)')
  return model.to(DEVICE)

def load_image(image_file_descriptor, model_settings):
  X = np.array(Image.open(image_file_descriptor))
  match model_settings['xnorm_type']:
    case 'imagenet': 
      m, s = [np.array(model_settings['xnorm_params'][k], dtype=np.float32)  * 255   for k in ('mean', 'std')]
    case 'image_per_channel':  
      m, s = X.mean(axis=(0,1)), X.std(axis=(0,1))
    case other: raise ValueError(f"Unknown xnorm_type '{other}' in model.settings")
  X = ((X - m) / s).transpose(2, 0, 1)
  return X,m,s


def count(images:list, model=None, plot=True):
  if model is None or type(model) == str:  
    model = init_model(model)
  model.eval()  # important

  counts = {}; plots = {}
  X,M,S = zip(*(load_image(i, model.settings) for i in images))
  X = np.stack(X)
  if os.uname().nodename == 'eli': X=X[:,:,:256,:256]  # NOTE for development on laptop. TODO implement tiled inference
  Y = model(torch.tensor(X).float().to(DEVICE)).detach().cpu().numpy()

  for i,image in enumerate(images):
    counts[image.name] = np.sum(Y[i])*model.settings["ymax"]
    if plot: plots[image.name] = plotting.overlay(X[i].transpose(1,2,0)*S[i]+M[i], Y[i], args_image={'norm':False})

  return counts, plots


if __name__ == '__main__':
  model_version = sys.argv[1] 
  targets = sys.argv[2:]

  # if target is a folder 
  image_paths = []
  for target in targets:
    if os.path.isdir(target):
      image_paths += [f"{target}/{f}" for f in os.listdir(target) if f.split('.')[-1] in ['jpg', 'jpeg', 'png', 'tiff']]
    elif os.path.isfile(target):
      image_paths += [target]
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
 
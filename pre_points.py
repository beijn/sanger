# %% 

import os 
from cellnet.data import *
from collections import defaultdict


label2int = {'Live Cell':1, 'Cell':1, 'cell':1, 'Dead cell/debris':2, 'Debris':2, 'debris': 2}

point_paths = ls('./data', 'json')
image_paths = ls('./data/images')

def load_points_json(paths, image_paths, l2i):
  image_ids = [imgid(p) for p in image_paths]
  P = defaultdict(lambda: defaultdict(list))
  for path in paths:
    a = os.path.basename(path).split('..')[-1].split('.')[0]
    for entry in (raw := json.load(open(path))):
      image_id = entry['data']['img'].split('/')[-1][9:].split('.')[0]
      if image_id not in image_ids: continue
      i = image_paths[image_ids.index(image_id)]
      for annotation in entry['annotations']:
        for result in annotation['result']: 
          x = result['value']['x']/100 * result['original_width']
          y = result['value']['y']/100 * result['original_height']
          l = result['value']['keypointlabels'][0].strip()
          if l not in l2i: raise ValueError(f"Label `{l}` from `{path}` not in mapping {l2i}")
          P[i][a] += [(x,y, l2i[l])]
  return mapd(P, lambda w: mapd(w,np.array)) 


def process_points(point_paths=point_paths, image_paths=image_paths, label2int=label2int):
  P = load_points_json(point_paths, image_paths, label2int)

  annotators = defaultdict(list)
  for i in image_paths:
    if i not in P: 
      print(f"WARN: No annotations for {i}")
      continue
    if len(P[i]) > 1: 
      print(f"WARN: Multiple annotations for {i}. TODO: fix disagreement")
      continue
    
    annotators[imgid(i)] += list(P[i].keys())
    np.save(f'data/cache/points/{imgid(i)}.npy', P[i][list(P[i].keys())[0]])

  json.dump(annotators, open('data/cache/annotators.json', 'w'), indent=2)
  return lambda: {i: np.load(f'data/cache/points/{imgid(i)}.npy') for i in P.keys()}, annotators

if __name__=='__main__': process_points()
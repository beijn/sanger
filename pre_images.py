# %%
from cellnet.data import *
import pandas as pd

image_paths = ls('./data/images')

# %%

def cache_fg_masks_using_stens_model(image_paths=image_paths, thresh=0.01):
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
  
  # if data/cache/fgmasks/{i}.npy exists, load it, otherwise create it
  X = load_images(image_paths)
  dir = os.path.expanduser('data/cache/fgmasks')
  os.makedirs(dir, exist_ok=True)
  M = {}
  for i in image_paths:
    if os.path.exists(f'{dir}/{imgid(i)}.npy'): 
      M[i] = np.load(f'{dir}/{imgid(i)}.npy')
      print(f"Loaded fgmask for {(i)}")
    else:
      M[i] = create_masks(X[i][None])[0]
      np.save(f'{dir}/{imgid(i)}.npy', M[i])
      print(f"Saved fgmask for {(i)}")
  return M

if __name__=='__main__': 
  M = cache_fg_masks_using_stens_model()

  from cellnet import plot 
  from cellnet.data import load_images, load_points
  import matplotlib.pyplot as plt

  for i in image_paths:
    x = load_images([i])[i]
    try: p = load_points([i])[i]
    except: p = np.zeros((0,3))
    m = M[i]

    ax = plot.overlay(x, None, M[i], p[:,[0,1]], p[:,2])
    # put text of the image id on top
    ax.text(x.shape[1]/2, x.shape[0]/2, imgid(i), ha='center', va='center', fontsize=5000, color='#00ff00')

    try: 
      get_ipython # type:ignore
    except: 
      plot.save(ax, f'mask-{imgid(i)}.png')
      plt.close(ax.figure) # type:ignore

  print('Please manually check the masks and update data/data_quality.csv')
  data_quality = pd.read_csv('data/data_quality.csv', sep=r'\s+', index_col=0)
  data_quality # type:ignore

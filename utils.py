import os
import json
import random
from time import time
from enum import Enum
from pathlib import Path
from PIL import Image, ImageFilter
from argparse import ArgumentParser, Namespace
import gc
from typing import *
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
import matplotlib.pyplot as plt

if 'repo':
  BASE_PATH = Path(__file__).parent.absolute()
  REPO_PATH = BASE_PATH / 'repo'
  SAM_PATH = REPO_PATH / 'segment-anything'
  SAM_CKPT_PATH = SAM_PATH / 'ckpt'
  SAM_DEMO_FILE = SAM_PATH / 'notebooks' / 'images' / 'dog.jpg'
  SAM_CKPTS = {
    'vit_b': 'sam_vit_b_01ec64.pth',
    'vit_l': 'sam_vit_l_0b3195.pth',
    'vit_h': 'sam_vit_h_4b8939.pth',
  }
  import sys
  sys.path.append(str(SAM_PATH))
  from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
  from segment_anything.modeling import Sam

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_PATH = Path(__file__).parent.absolute()
DATA_PATH = BASE_PATH / 'data'
DATASET_PATH = {
  'sam':   DATA_PATH / 'SAM_data',
}
OUT_PATH = BASE_PATH / 'out' ; OUT_PATH.mkdir(exist_ok=True)
npimg_u8  = NDArray[np.uint8]
npimg_u16 = NDArray[np.uint16]
npimg_f32 = NDArray[np.float32]
npimg_b1  = NDArray[np.bool_]
npimg     = Union[npimg_u8, npimg_f32, npimg_b1]
Data      = Union[ndarray, Tensor]
Size      = Tuple[int, int]
Point     = Tuple[int, int]
Prompts   = Tuple[ndarray, ndarray, None, None]

def seed_everything(seed:int):
  print('>> global seed:', seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

def timer(fn):
  def wrapper(*args, **kwargs):
    start = time()
    r = fn(*args, **kwargs)
    end = time()
    print(f'[Timer]: {fn.__name__} took {end - start:.3f}s')
    return r
  return wrapper

def get_all_tensors() -> List[Tensor]:
  tensors = []
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        tensors.append(obj)
    except:
      pass
  return tensors

def load_img(fp:Path, mode='RGB', dtype=np.uint8) -> Union[npimg_u8, npimg_f32]:
  assert dtype in [np.uint8, np.float32], 'invalid dtype, should be in [np.uint8, np.float32]'

  img = Image.open(str(fp)).convert(mode)
  im = np.array(img, dtype=np.uint8)
  if dtype is np.uint8: return im
  return (im / 255.0).astype(np.float32)

def show_img(im:npimg, figsize=(8, 6), anns:Dict[str, Any]=None):
  plt.figure(figsize=figsize)
  plt.imshow(im)
  if anns: show_anns(anns)
  plt.axis('off')
  plt.show()

def get_edge(im:npimg_u8) -> npimg_f32:
  assert is_npimg_u8(im), 'expect npimg of np.uint8'
  img = Image.fromarray(im).convert('RGB').filter(ImageFilter.FIND_EDGES).convert('L')
  return np.asarray(img, dtype=np.float32) / 255.

def make_diff(img:npimg_u8, adv:npimg_u8) -> npimg_f32:
  im0 = img / 255.0
  im1 = adv / 255.0
  d: npimg_f32 = np.abs(im0 - im1)
  print('Linf (proc):', d.max())
  print('L1 (proc):', d.mean())
  diff = minmax_norm(d)
  return diff

def img_to_red(im:npimg_u8, shift:int=32) -> npimg_u8:
  im = np.zeros_like(im)
  im[:, :, 0] = 255 - shift
  im[:, :, 1] = min(shift * 2, 255)
  im[:, :, 2] = min(shift * 2, 255)
  return im

def img_to_grey(im:npimg_u8) -> npimg_u8:
  img = Image.fromarray(im).convert('L')
  im = np.asarray(img, dtype=np.uint8)
  return np.expand_dims(im, -1)

def is_npimg_u8(im:npimg) -> bool:
  if not isinstance(im, ndarray): return False
  if im.dtype != np.uint8: return False
  if len(im.shape) not in [2, 3]: return False
  if len(im.shape) == 3 and im.shape[-1] not in [1, 3]: return False
  return True

def minmax_norm(x:Data, vmax:float=None) -> Data:
  if vmax is None: vmax = x.max() 
  return (x - x.min()) / (vmax - x.min())

def info_t(x:Data, name:str='x'):
  print(f'{name}: shape={tuple(x.shape)}, dtype={x.dtype}')

def get_iou(x: np.ndarray, y: np.ndarray) -> float:
  while x.ndim < y.ndim:
    x = np.expand_dims(x, 0)
  while y.ndim < x.ndim:
    y = np.expand_dims(y, 0)
  if x.shape == y.shape:
    denominator = np.sum(np.logical_or(x, y))
    return np.sum(np.logical_and(x, y)) / denominator if denominator != 0 else 0.0
  else:
    min_shape = tuple(min(a, b) for a, b in zip(x.shape, y.shape))
    x = x[tuple(slice(0, s) for s in min_shape)]
    y = y[tuple(slice(0, s) for s in min_shape)]
    denominator = np.sum(np.logical_or(x, y))
    return np.sum(np.logical_and(x, y)) / denominator if denominator != 0 else 0

def get_iou_auto(x:Union[Data, List[Data]], y:Data) -> float:
  if isinstance(x, list):
    iou = max([get_iou(m, y) for m in x])
  else:
    iou = get_iou(x, y)
  return iou

def load_json(fp:Path, default:Any=dict) -> Dict:
  if not fp.exists():
    assert isinstance(default, Callable), '"default" should be a callable'
    return default()
  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def save_json(data:Any, fp:Path):
  def _cvt(v:Any) -> Any:
    if   isinstance(v, Path): return str(v)
    elif isinstance(v, Enum): return str(v)
    else: return v
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False, default=_cvt)
load_cfg = load_json

def load_sam(model:str) -> Sam:
  fp = SAM_CKPT_PATH / SAM_CKPTS[model]
  print(f'>> load weights from {fp}')
  return sam_model_registry[model](checkpoint=str(fp)).eval().to('cuda:0')

def get_param_cnt(model:nn.Module) -> int:
  return sum([p.numel() for p in model.parameters() if p.requires_grad])

def get_parser() -> ArgumentParser:
  parser = ArgumentParser()
  parser.add_argument('-D', choices=DATASET_PATH.keys(), help='dataset name')
  parser.add_argument('-f', default=SAM_DEMO_FILE, help='path to image file')
  return parser

def get_args(parser:ArgumentParser=None) -> Namespace:
  parser = parser or get_parser()
  args = parser.parse_args()
  if not args.D: assert Path(args.f).is_file(), f'>> {args.f} is not a file'
  return args

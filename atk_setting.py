import os
import torch
import math
from torch.nn import Module
from torch.autograd import Function
import pywt
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch = torch.device('cuda:0')

try:
  from moviepy.editor import ImageSequenceClip
  HAS_MOVIEPY = True
except ImportError:
  print('>> [warn] missing lib "moviepy", will not generate adv pred step by step')
  HAS_MOVIEPY = False
from utils import *
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide

CAM_METH = ['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad']
LIM      = ['', 'edge', 'smap', 'cam', 'tgt']
SAM_MASK_THRESH = 0.0

class SamForwarder(nn.Module):
  def __init__(self, sam:Sam):
    super().__init__()
    assert SAM_MASK_THRESH == sam.mask_threshold, f'sam.mask_threshold ({sam.mask_threshold}) != SAM_MASK_THRESH ({SAM_MASK_THRESH})'
    self.model = sam
    self.canvas_size = sam.image_encoder.img_size
    self.device = sam.device
    self.transform = ResizeLongestSide(sam.image_encoder.img_size)

  def norm_image(self, x:Tensor) -> Tensor:
    return (x - self.model.pixel_mean) / self.model.pixel_std

  def denorm_image(self, x:Tensor) -> Tensor:
    return x * self.model.pixel_std + self.model.pixel_mean

  def resize_image(self, x:Tensor) -> Tensor:
    h, w = x.shape[-2:]
    return F.pad(x, (0, self.canvas_size - w, 0, self.canvas_size - h))

  def unresize_image(self, x:Tensor) -> Tensor:
    INTERP_MODE = 'bilinear'
    align_corners = None if INTERP_MODE == 'nearest' else False
    x = F.interpolate(x, (self.canvas_size, self.canvas_size), mode=INTERP_MODE, align_corners=align_corners)
    x = x[..., :self.input_size[0], :self.input_size[1]]
    return F.interpolate(x, self.original_size, mode=INTERP_MODE, align_corners=align_corners)

  def transform_image(self, im:npimg_u8, is_edge:bool=False) -> Tensor:
    x = self.transform.apply_image(im)
    X: Tensor = torch.from_numpy(x).to(self.device)
    X = X.permute(2, 0, 1).contiguous().unsqueeze_(0)
    assert (
      len(X.shape) == 4 and X.shape[1] == 3 and max(*X.shape[2:]) == self.canvas_size
    ), f"set_torch_image input must be BCHW with long side {self.canvas_size}."
    if not is_edge:
      self.original_size = im.shape[:2]
      self.input_size = tuple(X.shape[-2:])
      X = self.norm_image(X)
    return self.resize_image(X)

  def transform_prompts(self, point_coords:ndarray=None, point_labels:ndarray=None, box:ndarray=None, mask_input:ndarray=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
    if point_coords is not None:
      assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
      point_coords = self.transform.apply_coords(point_coords, self.original_size)
      coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
      labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
      coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    if box is not None:
      box = self.transform.apply_boxes(box, self.original_size)
      box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
      box_torch = box_torch[None, :]
    if mask_input is not None:
      mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
      mask_input_torch = mask_input_torch[None, :, :, :]
    return coords_torch, labels_torch, box_torch, mask_input_torch

  def forward(self, image:Tensor, point_coords:Tensor=None, point_labels:Tensor=None, boxes:Tensor=None, mask_input:Tensor=None, multi_mask:bool=False) -> Tuple[Tensor, Tensor]:
    features = self.model.image_encoder(image)
    self.features = self.model.image_encoder(image)
    points = (point_coords, point_labels) if point_coords is not None else None
    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points, boxes, mask_input)
    low_res_masks, iou_predictions = self.model.mask_decoder(
      image_embeddings=self.features,
      image_pe=self.model.prompt_encoder.get_dense_pe(),
      sparse_prompt_embeddings=sparse_embeddings,
      dense_prompt_embeddings=dense_embeddings,
      multimask_output=multi_mask,
    )
    masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
    return masks[0], iou_predictions[0]

FwdPack = Tuple[SamForwarder, Prompts, Callable]
PtorPack = Tuple[SamPredictor, Prompts]

def _parse_point(coord:str, size:Size) -> Point:
  if coord:
    point = list(reversed([float(e) for e in coord.split(',')]))
    for i, e in enumerate(point):
      if e < 1.0: point[i] = e * size[i]
      point[i] = int(point[i])
  else:
    point = [random.randrange(sz) for sz in size]
  return point

def make_prompts_ten(point: Union[str, ndarray], img_size: tuple, num_points: int = 10) -> Prompts:
  if isinstance(point, str) or point is None:
    point = _parse_point(point, img_size)
  coords = []
  for _ in range(num_points):
    random_point = generate_random_point(img_size)
    coords.append(random_point)
  coords = np.asarray(coords, dtype=np.int32)
  labels = np.ones(num_points, dtype=np.int32)
  return (coords, labels, None, None)

def generate_random_point(img_size: tuple):
  random_point = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
  return random_point

def make_prompts(point:Union[str, ndarray], img_size:tuple) -> Prompts:
  if isinstance(point, str) or point is None:
    point = _parse_point(point, img_size)
    coords = np.expand_dims(np.asarray(point, dtype=np.int32), axis=0)
  else:
    coords = point
  labels = np.asarray([1], dtype=np.int32)
  return (coords, labels, None, None)

def make_prompts_box(box: Union[str, ndarray], img_size: tuple) -> Prompts:
  box_array = np.asarray(box, dtype=np.int32)
  labels = np.asarray([1], dtype=np.int32)
  return (None, labels, box_array, None)

def make_prompts_randombox(box: Union[str, ndarray],img_size: tuple) -> Prompts:
  boxes = []
  y1, x1 = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
  y2, x2 = np.random.randint(y1, img_size[0]), np.random.randint(x1, img_size[1])
  boxes.append((y1, x1, y2, x2))
  box_array = np.expand_dims(np.asarray(boxes, dtype=np.int32), axis=0)
  labels = np.asarray([1], dtype=np.int32)
  return (None, labels, box_array, None)

def make_pred(ptor_pack:PtorPack, img:npimg_u8, multi_mask:bool=False, ret_logits:bool=False) -> Union[Tuple[npimg_b1, float], Tuple[List[npimg_b1], List[float]]]:
  ptor, prompts = ptor_pack
  ptor.set_image(img)
  mask, piou, _ = ptor.predict(*prompts, multimask_output=multi_mask, return_logits=ret_logits)
  ptor.reset_image()
  if multi_mask:
    return [mask[i] for i in range(len(mask))], piou.tolist()
  else:
    return mask[0], piou.item()

def make_tgt(ptor:SamPredictor, img:npimg_u8, point_tgt:Union[str, Point])-> npimg_b1:
  if not point_tgt: return None
  img_size = img.shape[:-1]
  prompts_tgt = make_prompts(point_tgt, img_size)
  tgt, _ = make_pred((ptor, prompts_tgt), img)
  return tgt

def make_Y(tgt:npimg_b1=None, img:npimg_u8=None, w:float=-10.0) -> Tensor:
  '''
    non-targeted: (img.shape, loss_w) -> Tensor
    targeted: npimg_b1 -> Tensor
  '''

  if tgt is not None:
    Y = torch.from_numpy(tgt).float()
  else:
    H, W, _ = img.shape
    Y = torch.ones([H, W]) * w
  return Y.unsqueeze_(0)

@torch.no_grad()
def _make_smap(args, fwd_pack:FwdPack, img:npimg_u8, tgt:npimg_b1) -> npimg_f32:
  decode = lambda x: fwder.unresize_image(x)[0].permute(1, 2, 0).cpu().numpy()
  fwder, prompts, loss_fn = fwd_pack
  X = fwder.transform_image(img)
  P = fwder.transform_prompts(*prompts)
  Y = make_Y(tgt, img, args.loss_w).to(X.device)

  with torch.enable_grad():
    X.requires_grad = True
    logits, piou = fwder.forward(X, *P)
    loss = loss_fn(logits, Y)

  g = grad(loss, X, loss)[0]
  psmap = F.tanh(g.abs())
  psmap /= psmap.max()
  smap: npimg_f32 = decode(psmap)
  smap = smap.mean(axis=-1)
  return smap

def get_parser() -> ArgumentParser:
  from utils import get_parser as get_base_parser
  parser = get_base_parser()
  parser.add_argument('--cam_meth', default='GradCAM', choices=CAM_METH, help='cam method')
  parser.add_argument('--debug', action='store_true', help='show detailed pgd log by step')
  return parser

def get_args(parser:ArgumentParser) -> Namespace:
  from utils import get_args as get_base_args
  args = get_base_args(parser)
  seed_everything(args.seed)
  args.debug = True
  return args

if __name__ == '__main__':
  parser = get_parser()
  parser.add_argument('--point',     help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
  parser.add_argument('--point_tgt', help='alike --point, but specify target mask point to run targeted attack; default image is -f')
  parser.add_argument('--f_tgt',     help='alike -f, but specify target image filepath, change image of --point_tgt')
  args = get_args(parser)


  class DWTFunction_2D_tiny(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
      ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
      matrix_Low_0 = matrix_Low_0.to(device)
      matrix_Low_1 = matrix_Low_1.to(device)
      L = torch.matmul(matrix_Low_0, input)
      LL = torch.matmul(L, matrix_Low_1)
      return LL

    @staticmethod
    def backward(ctx, grad_LL):
      matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
      matrix_Low_0 = matrix_Low_0.to(device)
      matrix_Low_1 = matrix_Low_1.to(device)
      grad_L = torch.matmul(grad_LL, matrix_Low_1.t())
      grad_input = torch.matmul(matrix_Low_0.t(), grad_L)
      return grad_input, None, None, None, None


class IDWT_2D_tiny(Module):
  """
  input:  lfc -- (N, C, H/2, W/2)
          hfc_lh -- (N, C, H/2, W/2)
          hfc_hl -- (N, C, H/2, W/2)
          hfc_hh -- (N, C, H/2, W/2)
  output: the original 2D data -- (N, C, H, W)
  """

  def __init__(self, wavename):
    """
    2D inverse DWT (IDWT) for 2D image reconstruction
    :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
    """
    super(IDWT_2D_tiny, self).__init__()
    wavelet = pywt.Wavelet(wavename)
    self.band_low = wavelet.dec_lo
    self.band_low.reverse()
    self.band_high = wavelet.dec_hi
    self.band_high.reverse()
    assert len(self.band_low) == len(self.band_high)
    self.band_length = len(self.band_low)
    assert self.band_length % 2 == 0
    self.band_length_half = math.floor(self.band_length / 2)

  def get_matrix(self):
    L1 = np.max((self.input_height, self.input_width))
    L = math.floor(L1 / 2)
    matrix_h = np.zeros((L, L1 + self.band_length - 2))
    matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
    end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

    index = 0
    for i in range(L):
      for j in range(self.band_length):
        matrix_h[i, index + j] = self.band_low[j]
      index += 2
    matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
    matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

    index = 0
    for i in range(L1 - L):
      for j in range(self.band_length):
        matrix_g[i, index + j] = self.band_high[j]
      index += 2
    matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                 0:(self.input_height + self.band_length - 2)]
    matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                 0:(self.input_width + self.band_length - 2)]

    matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
    matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
    matrix_h_1 = np.transpose(matrix_h_1)
    matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
    matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
    matrix_g_1 = np.transpose(matrix_g_1)
    if torch.cuda.is_available():
      self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
      self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
      self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
      self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
    else:
      self.matrix_low_0 = torch.Tensor(matrix_h_0)
      self.matrix_low_1 = torch.Tensor(matrix_h_1)
      self.matrix_high_0 = torch.Tensor(matrix_g_0)
      self.matrix_high_1 = torch.Tensor(matrix_g_1)

  def forward(self, LL):
    """
    recontructing the original 2D data
    the original 2D data = \mathcal{L}^T * lfc * \mathcal{L}
                         + \mathcal{H}^T * hfc_lh * \mathcal{L}
                         + \mathcal{L}^T * hfc_hl * \mathcal{H}
                         + \mathcal{H}^T * hfc_hh * \mathcal{H}
    :param LL: the low-frequency component
    :param LH: the high-frequency component, hfc_lh
    :param HL: the high-frequency component, hfc_hl
    :param HH: the high-frequency component, hfc_hh
    :return: the original 2D data
    """
    assert len(LL.size()) == 4
    self.input_height = LL.size()[-2] * 2
    self.input_width = LL.size()[-1] * 2
    self.get_matrix()
    return IDWTFunction_2D_tiny.apply(LL, self.matrix_low_0, self.matrix_low_1)


class IDWT_2D_tiny_hh(Module):
  """
  input:  lfc -- (N, C, H/2, W/2)
          hfc_lh -- (N, C, H/2, W/2)
          hfc_hl -- (N, C, H/2, W/2)
          hfc_hh -- (N, C, H/2, W/2)
  output: the original 2D data -- (N, C, H, W)
  """

  def __init__(self, wavename):
    """
    2D inverse DWT (IDWT) for 2D image reconstruction
    :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
    """
    super(IDWT_2D_tiny_hh, self).__init__()
    wavelet = pywt.Wavelet(wavename)
    self.band_low = wavelet.dec_lo
    self.band_low.reverse()
    self.band_high = wavelet.dec_hi
    self.band_high.reverse()
    assert len(self.band_low) == len(self.band_high)
    self.band_length = len(self.band_low)
    assert self.band_length % 2 == 0
    self.band_length_half = math.floor(self.band_length / 2)

  def get_matrix(self):
    """
    generating the matrices: \mathcal{L}, \mathcal{H}
    :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
    """
    L1 = np.max((self.input_height, self.input_width))
    L = math.floor(L1 / 2)
    matrix_h = np.zeros((L, L1 + self.band_length - 2))
    matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
    end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

    index = 0
    for i in range(L):
      for j in range(self.band_length):
        matrix_h[i, index + j] = self.band_low[j]
      index += 2
    matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
    matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

    index = 0
    for i in range(L1 - L):
      for j in range(self.band_length):
        matrix_g[i, index + j] = self.band_high[j]
      index += 2
    matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                 0:(self.input_height + self.band_length - 2)]
    matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                 0:(self.input_width + self.band_length - 2)]

    matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
    matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
    matrix_h_1 = np.transpose(matrix_h_1)
    matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
    matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
    matrix_g_1 = np.transpose(matrix_g_1)
    if torch.cuda.is_available():
      self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
      self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
      self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
      self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
    else:
      self.matrix_low_0 = torch.Tensor(matrix_h_0)
      self.matrix_low_1 = torch.Tensor(matrix_h_1)
      self.matrix_high_0 = torch.Tensor(matrix_g_0)
      self.matrix_high_1 = torch.Tensor(matrix_g_1)

  def forward(self, HH):
    assert len(HH.size()) == 4
    self.input_height = HH.size()[-2] * 2
    self.input_width = HH.size()[-1] * 2
    self.get_matrix()
    return IDWTFunction_2D_tiny.apply(HH, self.matrix_high_0, self.matrix_high_1)


class DWT_2D_tiny(Module):
  """
  input: the 2D data to be decomposed -- (N, C, H, W)
  output -- lfc: (N, C, H/2, W/2)
            #hfc_lh: (N, C, H/2, W/2)
            #hfc_hl: (N, C, H/2, W/2)
            #hfc_hh: (N, C, H/2, W/2)
  DWT_2D_tiny only outputs the low-frequency component, which is used in WaveCNet;
  the all four components could be get using DWT_2D, which is used in WaveUNet.
  """

  def __init__(self, wavename):
    """
    2D discrete wavelet transform (DWT) for 2D image decomposition
    :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
    """
    super(DWT_2D_tiny, self).__init__()
    wavelet = pywt.Wavelet(wavename)
    self.band_low = wavelet.rec_lo
    self.band_high = wavelet.rec_hi
    assert len(self.band_low) == len(self.band_high)
    self.band_length = len(self.band_low)  # 2
    assert self.band_length % 2 == 0
    self.band_length_half = math.floor(self.band_length / 2)

  def get_matrix(self):
    """
    生成变换矩阵
    generating the matrices: \mathcal{L}, \mathcal{H}
    :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
    """
    L1 = np.max((self.input_height, self.input_width))  # 224
    L = math.floor(L1 / 2)  # 112
    matrix_h = np.zeros((L, L1 + self.band_length - 2))  # (112, 224 + 2 -2)
    matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
    end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

    index = 0
    for i in range(L):
      for j in range(self.band_length):
        matrix_h[i, index + j] = self.band_low[j]
      index += 2
    matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
    matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
    index = 0
    for i in range(L1 - L):
      for j in range(self.band_length):
        matrix_g[i, index + j] = self.band_high[j]
      index += 2
    matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                 0:(self.input_height + self.band_length - 2)]
    matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                 0:(self.input_width + self.band_length - 2)]
    matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
    matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
    matrix_h_1 = np.transpose(matrix_h_1)
    matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
    matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
    matrix_g_1 = np.transpose(matrix_g_1)

    if torch.cuda.is_available():
      self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
      self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
      self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
      self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
    else:
      self.matrix_low_0 = torch.Tensor(matrix_h_0)
      self.matrix_low_1 = torch.Tensor(matrix_h_1)
      self.matrix_high_0 = torch.Tensor(matrix_g_0)
      self.matrix_high_1 = torch.Tensor(matrix_g_1)

  def forward(self, input):
    """
    input_lfc = \mathcal{L} * input * \mathcal{L}^T
    #input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
    #input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
    #input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
    :param input: the 2D data to be decomposed
    :return: the low-frequency component of the input 2D data
    """
    assert len(input.size()) == 4
    self.input_height = input.size()[-2]
    self.input_width = input.size()[-1]
    self.get_matrix()
    return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0,
                                     self.matrix_high_1)


class IDWTFunction_2D_tiny_hh(Function):
  @staticmethod
  def forward(ctx, input_HH, matrix_High_0, matrix_High_1):
    ctx.save_for_backward(matrix_High_0, matrix_High_1)
    matrix_High_0 = matrix_High_0.to(device)
    matrix_High_1 = matrix_High_1.to(device)
    H = torch.matmul(input_HH, matrix_High_1.t())
    output = torch.matmul(matrix_High_0.t(), H)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    matrix_High_0, matrix_High_1 = ctx.saved_variables
    matrix_High_0 = matrix_High_0.to(device)
    matrix_High_1 = matrix_High_1.to(device)
    grad_H = torch.matmul(matrix_High_0, grad_output)
    grad_HH = torch.matmul(grad_H, matrix_High_1)
    return grad_HH, None, None


class IDWTFunction_2D_tiny(Function):
  @staticmethod
  def forward(ctx, input_LL, matrix_Low_0, matrix_Low_1):
    ctx.save_for_backward(matrix_Low_0, matrix_Low_1)
    matrix_Low_1 = matrix_Low_1.to(device)
    matrix_Low_0 = matrix_Low_0.to(device)
    L = torch.matmul(input_LL, matrix_Low_1.t())
    output = torch.matmul(matrix_Low_0.t(), L)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    matrix_Low_0, matrix_Low_1 = ctx.saved_variables
    matrix_Low_0 = matrix_Low_0.to(device)
    matrix_Low_1 = matrix_Low_1.to(device)
    grad_L = torch.matmul(matrix_Low_0, grad_output)
    grad_LL = torch.matmul(grad_L, matrix_Low_1)
    return grad_LL, None, None, None, None


class DWT_2D(Module):
  """
  input: the 2D data to be decomposed -- (N, C, H, W)
  output -- lfc: (N, C, H/2, W/2)
            hfc_lh: (N, C, H/2, W/2)
            hfc_hl: (N, C, H/2, W/2)
            hfc_hh: (N, C, H/2, W/2)
  """

  def __init__(self, wavename):
    """
    2D discrete wavelet transform (DWT) for 2D image decomposition
    :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
    """
    super(DWT_2D, self).__init__()
    wavelet = pywt.Wavelet(wavename)
    self.band_low = wavelet.rec_lo
    self.band_high = wavelet.rec_hi
    assert len(self.band_low) == len(self.band_high)
    self.band_length = len(self.band_low)
    assert self.band_length % 2 == 0
    self.band_length_half = math.floor(self.band_length / 2)

  def get_matrix(self):
    """
    生成变换矩阵
    generating the matrices: \mathcal{L}, \mathcal{H}
    :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
    """
    L1 = np.max((self.input_height, self.input_width))
    L = math.floor(L1 / 2)
    matrix_h = np.zeros((L, L1 + self.band_length - 2))
    matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
    end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

    index = 0
    for i in range(L):
      for j in range(self.band_length):
        matrix_h[i, index + j] = self.band_low[j]
      index += 2
    matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
    matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

    index = 0
    for i in range(L1 - L):
      for j in range(self.band_length):
        matrix_g[i, index + j] = self.band_high[j]
      index += 2
    matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                 0:(self.input_height + self.band_length - 2)]
    matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                 0:(self.input_width + self.band_length - 2)]

    matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
    matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
    matrix_h_1 = np.transpose(matrix_h_1)
    matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
    matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
    matrix_g_1 = np.transpose(matrix_g_1)

    if torch.cuda.is_available():
      self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
      self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
      self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
      self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
    else:
      self.matrix_low_0 = torch.Tensor(matrix_h_0)
      self.matrix_low_1 = torch.Tensor(matrix_h_1)
      self.matrix_high_0 = torch.Tensor(matrix_g_0)
      self.matrix_high_1 = torch.Tensor(matrix_g_1)

  def forward(self, input):
    """
    input_lfc = \mathcal{L} * input * \mathcal{L}^T
    input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
    input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
    input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
    :param input: the 2D data to be decomposed
    :return: the low-frequency and high-frequency components of the input 2D data
    """
    assert len(input.size()) == 4
    self.input_height = input.size()[-2]
    self.input_width = input.size()[-1]
    self.get_matrix()
    return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class IDWT_2D(Module):
  """
  input:  lfc -- (N, C, H/2, W/2)
          hfc_lh -- (N, C, H/2, W/2)
          hfc_hl -- (N, C, H/2, W/2)
          hfc_hh -- (N, C, H/2, W/2)
  output: the original 2D data -- (N, C, H, W)
  """

  def __init__(self, wavename):
    """
    2D inverse DWT (IDWT) for 2D image reconstruction
    :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
    """
    super(IDWT_2D, self).__init__()
    wavelet = pywt.Wavelet(wavename)
    self.band_low = wavelet.dec_lo
    self.band_low.reverse()
    self.band_high = wavelet.dec_hi
    self.band_high.reverse()
    assert len(self.band_low) == len(self.band_high)
    self.band_length = len(self.band_low)
    assert self.band_length % 2 == 0
    self.band_length_half = math.floor(self.band_length / 2)

  def get_matrix(self):
    """
    generating the matrices: \mathcal{L}, \mathcal{H}
    :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
    """
    L1 = np.max((self.input_height, self.input_width))
    L = math.floor(L1 / 2)
    matrix_h = np.zeros((L, L1 + self.band_length - 2))
    matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
    end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

    index = 0
    for i in range(L):
      for j in range(self.band_length):
        matrix_h[i, index + j] = self.band_low[j]
      index += 2
    matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
    matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

    index = 0
    for i in range(L1 - L):
      for j in range(self.band_length):
        matrix_g[i, index + j] = self.band_high[j]
      index += 2
    matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),
                 0:(self.input_height + self.band_length - 2)]
    matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),
                 0:(self.input_width + self.band_length - 2)]

    matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
    matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
    matrix_h_1 = np.transpose(matrix_h_1)
    matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
    matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
    matrix_g_1 = np.transpose(matrix_g_1)
    if torch.cuda.is_available():
      self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
      self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
      self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
      self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
    else:
      self.matrix_low_0 = torch.Tensor(matrix_h_0)
      self.matrix_low_1 = torch.Tensor(matrix_h_1)
      self.matrix_high_0 = torch.Tensor(matrix_g_0)
      self.matrix_high_1 = torch.Tensor(matrix_g_1)

  def forward(self, LL, LH, HL, HH):
    """
    recontructing the original 2D data
    the original 2D data = \mathcal{L}^T * lfc * \mathcal{L}
                         + \mathcal{H}^T * hfc_lh * \mathcal{L}
                         + \mathcal{L}^T * hfc_hl * \mathcal{H}
                         + \mathcal{H}^T * hfc_hh * \mathcal{H}
    :param LL: the low-frequency component
    :param LH: the high-frequency component, hfc_lh
    :param HL: the high-frequency component, hfc_hl
    :param HH: the high-frequency component, hfc_hh
    :return: the original 2D data
    """
    assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
    self.input_height = LL.size()[-2] + HH.size()[-2]
    self.input_width = LL.size()[-1] + HH.size()[-1]
    self.get_matrix()
    return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0,
                                 self.matrix_high_1)


class IDWTFunction_2D(Function):
  @staticmethod
  def forward(ctx, input_LL, input_LH, input_HL, input_HH,
              matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
    ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
    # L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
    matrix_Low_0 = matrix_Low_0.to(device)
    matrix_Low_1 = matrix_Low_1.to(device)
    matrix_High_0 = matrix_High_0.to(device)
    matrix_High_1 = matrix_High_1.to(device)

    L = torch.matmul(input_LH, matrix_High_1.t())
    H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
    output = torch.add(torch.matmul(matrix_Low_0.t(), L), torch.matmul(matrix_High_0.t(), H))
    return output

  @staticmethod
  def backward(ctx, grad_output):
    matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
    matrix_Low_0 = matrix_Low_0.to(device)
    matrix_Low_1 = matrix_Low_1.to(device)
    matrix_High_0 = matrix_High_0.to(device)
    matrix_High_1 = matrix_High_1.to(device)
    grad_L = torch.matmul(matrix_Low_0, grad_output)
    grad_H = torch.matmul(matrix_High_0, grad_output)
    grad_LL = torch.matmul(grad_L, matrix_Low_1)
    grad_LH = torch.matmul(grad_L, matrix_High_1)
    grad_HL = torch.matmul(grad_H, matrix_Low_1)
    grad_HH = torch.matmul(grad_H, matrix_High_1)
    return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None


class DWTFunction_2D(Function):
  @staticmethod
  def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
    ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
    matrix_Low_0 = matrix_Low_0.to(device)
    matrix_Low_1 = matrix_Low_1.to(device)
    matrix_High_0 = matrix_High_0.to(device)
    matrix_High_1 = matrix_High_1.to(device)
    L = torch.matmul(matrix_Low_0, input)
    H = torch.matmul(matrix_High_0, input)
    LL = torch.matmul(L, matrix_Low_1)
    LH = torch.matmul(L, matrix_High_1)
    HL = torch.matmul(H, matrix_Low_1)
    HH = torch.matmul(H, matrix_High_1)
    return LL, LH, HL, HH

  @staticmethod
  def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
    matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
    matrix_Low_0 = matrix_Low_0.to(device)
    matrix_Low_1 = matrix_Low_1.to(device)
    matrix_High_0 = matrix_High_0.to(device)
    matrix_High_1 = matrix_High_1.to(device)

    grad_L = torch.matmul(grad_LH, matrix_High_1.t())
    grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
    grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
    return grad_input, None, None, None, None
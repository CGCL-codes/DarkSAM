from atk_setting import *
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image

try:
    from pycocotools.mask import decode
except ImportError:
    print('>> [error] missing lib "pycocotools", run "pip install pycocotools" first!!')
    raise

def resize_mask_and_box(mask_gt, old_box_ann, target_size):
    box_ann =[old_box_ann[0],old_box_ann[1],old_box_ann[0]+old_box_ann[2],old_box_ann[1]+old_box_ann[3]]
    original_size = (mask_gt.shape[0], mask_gt.shape[1])
    scale_factor_x = target_size[1] / original_size[1]
    scale_factor_y = target_size[0] / original_size[0]
    resized_mask = Image.fromarray(mask_gt)
    resized_mask = resized_mask.resize(target_size[::-1], Image.BILINEAR)
    resized_mask = np.array(resized_mask)
    new_box_ann = [
        int(box_ann[0] * scale_factor_x),
        int(box_ann[1] * scale_factor_y),
        int(box_ann[2] * scale_factor_x),
        int(box_ann[3] * scale_factor_y)
    ]
    return resized_mask, new_box_ann

def resize_mask(mask_gt, target_size, center_x, center_y):
    original_size = (mask_gt.shape[0], mask_gt.shape[1])
    scale_factor_x = target_size[1] / original_size[1]
    scale_factor_y = target_size[0] / original_size[0]
    resized_mask = Image.fromarray(mask_gt)
    resized_mask = resized_mask.resize(target_size[::-1], Image.BILINEAR)
    resized_mask = np.array(resized_mask)
    new_center_x = int(center_x * scale_factor_x)
    new_center_y = int(center_y * scale_factor_y)
    return resized_mask, new_center_x, new_center_y

class Dataset_SA1B(Dataset):

    def __init__(self, sample_ids, data_root, target_size=(1500, 2250), do_shuffle=True, args = None):
        self.sample_ids = sample_ids
        self.data_root = data_root
        self.target_size = target_size
        self.do_shuffle = do_shuffle
        self.args = args
        self.sam = load_sam(args.M)
        combined_data = list(zip(self.sample_ids, self._load_data()))
        if self.do_shuffle:
            combined_data = shuffle(combined_data)
        self.sample_ids, self.data = zip(*combined_data)
    def _load_data(self):
        data = []
        failed_samples = []
        for sample_id in self.sample_ids:
            image_path = self.data_root / f"{sample_id}.jpg"
            try:
                image = Image.open(image_path).convert("RGB")
                image = np.array(image.resize(self.target_size[::-1], Image.BILINEAR))
                cfg = load_cfg(self.data_root / f'{sample_id}.json')
                annots = cfg['annotations']
                annots_sel = np.random.choice(annots, size=1, replace=False) if 0 < 1 < len(
                    annots) else annots
                sam_fwder = SamForwarder(self.sam)
                X = sam_fwder.transform_image(image)
                for annot in annots_sel:
                    mask_gt: npimg_b1 = np.ascontiguousarray(decode(annot['segmentation']), dtype=bool)
                    if self.args.test_prompts == 'pt':
                        print("pt")
                        point_ann = np.asarray(annot['point_coords'])
                        center_x = point_ann[0, 0]
                        center_y = point_ann[0, 1]
                        resized_mask_gt, resized_center_x, resized_center_y = resize_mask(mask_gt,
                                                                                          target_size=self.target_size,
                                                                                          center_x=center_x,
                                                                                          center_y=center_y)
                        point = np.array([[resized_center_x, resized_center_y]])
                        prompts = make_prompts(point, image)
                        P = sam_fwder.transform_prompts(*prompts)
                    if self.args.test_prompts == 'bx':
                        print("bx")
                        box_ann = np.asarray(annot['bbox'])
                        resized_mask_gt, new_box_ann = resize_mask_and_box(mask_gt, box_ann, self.target_size)
                        prompts = make_prompts_box(new_box_ann, image)
                        P = sam_fwder.transform_prompts(*prompts)
                data.append((image, P, resized_mask_gt, sample_id))
                print(f"from SA1B Processing sample {sample_id}...")
            except Exception as e:
                print(f"Error loading data for sample {sample_id}: {str(e)}")
                failed_samples.append(sample_id)
                continue
        return data
    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        return self.data[idx]

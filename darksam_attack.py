from dataset import *
from atk_setting import *
import torch.nn.functional as F
from torch.autograd import grad
from tqdm import tqdm
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
device_list = [int(device_idx) for device_idx in cuda_visible_devices.split(',')]
print(f'CUDA_VISIBLE_DEVICES: {device_list}')
DATA_ROOT = DATASET_PATH['sam']
HIST_FILE = OUT_PATH / 'atk_sam.json'

try:
    from pycocotools.mask import decode
except ImportError:
    print('>> [error] missing lib "pycocotools", run "pip install pycocotools" first!!')
    raise

def get_args(parser: ArgumentParser) -> Namespace:
    from atk_setting import get_args as get_base_args
    args = get_base_args(parser)
    args.f = None
    args.D = 'sam'
    args.fps = -1
    args.debug = False
    return args

def choose_dataset(dataset,args = None):
    print("args.sta:",args.sta)
    if dataset == 'SA1B':
        sample_ids = sorted({fp.stem for fp in DATA_ROOT.iterdir()})
        np.random.shuffle(sample_ids)
        if args.limit_img > 0:
            if args.sta == 'train':
                sample_ids = sample_ids[:args.limit_img]
            if args.sta == 'test':
                sample_ids = sample_ids[-args.limit_img:][::-1]
        custom_dataset = Dataset_SA1B(sample_ids, DATA_ROOT, args=args)
    return custom_dataset

def collate_fn(batch):
    images, mask_gt_list, prompts_list, sample_id = zip(*batch)
    return images, mask_gt_list, prompts_list, sample_id

def run(args, custom_dataset):
    sam = load_sam(args.M)
    sam = sam.to(device)
    sam_fwder = SamForwarder(sam)
    loss_fn = F.mse_loss
    norm = lambda x: sam_fwder.norm_image(x * 255)
    batch_size = 1
    alpha = 1/255
    epsilon = args.eps / 255
    tensor_shape = (1, 3, 1024, 1024)
    shape_tensor = torch.empty(tensor_shape)
    perturbation = torch.empty_like(shape_tensor).uniform_(-epsilon, epsilon)
    perturbation = perturbation.to(device)
    lmbd = 0.1
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
    image_id = 0
    weight_Y = args.Y
    P_num = args.P_num

    for images, P_list, Mask_gt_list, img_ids in tqdm(data_loader):
        img_ID = img_ids[0]
        print("img_ID:",img_ID)
        image_id += 1
        img = images[0]
        denorm = lambda x: sam_fwder.denorm_image(x) / 255.0
        _cvt = lambda x: torch.as_tensor(x).round().clamp(0, 255).byte().permute(1, 2, 0).cuda().numpy()
        X = sam_fwder.transform_image(img)
        X = X.to(device)
        benign_img = (denorm(X) * 255).byte().div(255)
        H, W, _ = img.shape
        Y = torch.ones([1, H, W]).to(X.device, torch.float32) * weight_Y
        Y_bin = Y.bool()
        assert Y_bin.dtype in ['bool', bool, torch.bool]
        print(f"args.train_dataset: {args.train_dataset} ")

        for step in range(0, P_num):
            adv_img = benign_img + perturbation
            adv_img.requires_grad = True
            if args.train_prompts == 'pt':
                prompts = make_prompts(args.point, img.shape[:-1])
            if args.train_prompts == 'bx':
                prompts = make_prompts_randombox(args.point, img.shape[:-1])
            P = sam_fwder.transform_prompts(*prompts)
            logits, _ = sam_fwder.forward(adv_img, *P)
            mask = logits > sam_fwder.model.mask_threshold
            attacked = mask[0] == Y_bin
            output = attacked * logits
            output_f = ~attacked * logits
            loss_t = loss_fn(output, Y)
            loss_f = loss_fn(output_f, -Y)
            wave: str = 'haar'
            DWT = DWT_2D(wavename=wave)
            IDWT = IDWT_2D(wavename=wave)
            IDWT_ll = IDWT_2D_tiny(wavename=wave)
            Fre_loss = nn.MSELoss(reduction='sum')
            inputs_ll, inputs_hl, inputs_lh, inputs_hh = DWT(benign_img)
            adv_ll, adv_hl, adv_lh, adv_hh = DWT(adv_img)
            input_img_ll = IDWT_ll(inputs_ll)
            adv_img_ll = IDWT_ll(adv_ll)
            input_img_hh = IDWT(torch.zeros_like(inputs_ll), torch.zeros_like(inputs_hl), torch.zeros_like(inputs_lh), inputs_hh)
            adv_img_hh = IDWT(torch.zeros_like(adv_ll), torch.zeros_like(adv_hl), torch.zeros_like(adv_lh), adv_hh)
            low_freq_loss_ll = Fre_loss(adv_img_ll, input_img_ll)
            high_freq_loss_hh = -Fre_loss(adv_img_hh, input_img_hh)
            loss = loss_t * args.Wt + loss_f * args.Wf + low_freq_loss_ll * args.Wl + high_freq_loss_hh * args.Wh
            g = grad(loss, adv_img, loss)[0]
            delta = g.sign() * alpha
            perturbation = perturbation + delta
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            perturbation = perturbation.detach()
        del X, benign_img, Y, adv_img, delta
        if image_id == args.train_num:
            uap = perturbation
            uap_save_path = f"uap_file/{args.train_dataset}.pth"
            if not os.path.exists(os.path.dirname(uap_save_path)):
                os.makedirs(os.path.dirname(uap_save_path), exist_ok=True)
                print(f"Directory '{os.path.dirname(uap_save_path)}' created.")
            torch.save(uap, uap_save_path)
            print("uap saved in :", uap_save_path)

def get_parser() -> ArgumentParser:
    from atk_setting import get_parser as get_base_parser
    parser = get_base_parser()
    parser.add_argument('--limit_img', default=105, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--point', help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
    parser.add_argument('--sta', choices=['train', 'test'], default='train', help='station of comm')
    parser.add_argument('--train_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--test_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--Y', default=-1, type=float, help='weight for weight_Y')
    parser.add_argument('--seed', default=100, type=int, help='rand seed')
    parser.add_argument('-d', '--train_dataset',default='SA1B')
    parser.add_argument('--eps', default=10, type=float)
    parser.add_argument('--P_num', default=10, type=float)
    parser.add_argument('--train_num', default=100, type=float)
    parser.add_argument('--Wt', default=1, type=float, help='weight for t')
    parser.add_argument('--Wf', default=0.1, type=float, help='weight for f')
    parser.add_argument('--Wl', default=1, type=float, help='weight for l')
    parser.add_argument('--Wh', default=0.01, type=float, help='weight for h')
    parser.add_argument('--M', default='vit_b', choices=SAM_CKPTS.keys(), help='model checkpoint')
    return parser

if __name__ == '__main__':
    device =('cuda:0')
    parser = get_parser()
    args = get_args(parser)
    custom_dataset = choose_dataset(args.train_dataset, args)
    run(args, custom_dataset)











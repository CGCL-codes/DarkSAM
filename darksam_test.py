import csv
from darksam_attack import choose_dataset
from atk_setting import *
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm

try:
    from pycocotools.mask import decode
except ImportError:
    print('>> [error] missing lib "pycocotools", run "pip install pycocotools" first!!')
    raise
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
DATA_ROOT = DATASET_PATH['sam']
HIST_FILE = OUT_PATH / 'atk_sam.json'
transform = transforms.Compose([transforms.ToTensor()])

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    fileName = datetime.now().strftime('%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    return fileName

def collate_fn(batch):
    images, prompts_list, mask_gt_list, sample_id  = zip(*batch)
    return images, prompts_list, mask_gt_list, sample_id

def run(args,path_id, custom_dataset):
    sam = load_sam(args.M)
    sam = sam.to(device)
    sam_fwder = SamForwarder(sam)
    sam_fwder = sam_fwder.to(device)
    loss_fn = F.mse_loss
    s = time()
    iou_sum_adv, iou_cnt = 0.0, 0
    iou_sum_img = 0.0
    interrupted = False
    batch_size = 1
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
    denorm = lambda x: sam_fwder.denorm_image(x) / 255.0
    uap_save_path = path_id
    uap = torch.load(uap_save_path, map_location=device)
    try:
        for batch in tqdm(data_loader):
            images, p_list, mask_gt_list, sample_ids = batch
            mask_gt = mask_gt_list[0]
            P = p_list[0]
            for image in images:
                X = sam_fwder.transform_image(image)
                X = X.to(device)
                benign_img = (denorm(X) * 255).byte().div(255)
                benign_img = benign_img.to(device)
                adv_img = benign_img + uap
                adv_img = torch.clamp(adv_img, 0, 1)
                with torch.no_grad():
                    logits_clean, _ = sam_fwder.forward(benign_img, *P)
                logits_clean = logits_clean.to(device)
                mask_clean = logits_clean > sam_fwder.model.mask_threshold
                mask_clean = mask_clean.to(device)
                with torch.no_grad():
                    logits_hat, _ = sam_fwder.forward(adv_img, *P)
                logits_hat = logits_hat.to(device)
                mask_hat = logits_hat > sam_fwder.model.mask_threshold
                mask_hat = mask_hat.to(device)
                iou_benign_img = get_iou_auto(mask_clean.cpu().detach().numpy(), mask_gt)
                if iou_benign_img >=args.clean_lim:
                    iou_adv_img = get_iou_auto(mask_hat.cpu().detach().numpy(), mask_gt)
                    iou_benign_img_percentage = iou_benign_img * 100
                    iou_adv_img_percentage = iou_adv_img * 100
                    print(f"iou_benign_img: {iou_benign_img_percentage:.2f} %,  iou_adv_img:, {iou_adv_img_percentage:.2f} %")
                    iou_cnt += 1
                    iou_sum_img = iou_sum_img + iou_benign_img
                    iou_sum_adv = iou_sum_adv + iou_adv_img
                del logits_clean, benign_img, X, mask_clean, logits_hat, mask_hat, adv_img
                if iou_cnt == args.test_num:
                    exit()
    except KeyboardInterrupt:
        print('>> interrupted!!')
        interrupted = True
    except:
        print("exit")
        exit()
    finally:
        miou_img = 0.0 if iou_cnt == 0 else (iou_sum_img / iou_cnt)
        miou_adv = 0.0 if iou_cnt == 0 else (iou_sum_adv / iou_cnt)
        print(f'>> miou_benign_img: {miou_img}, >> miou_adv_img: {miou_adv}, >> iou_cnt:{iou_cnt}')
        return iou_cnt,miou_img, miou_adv
        t = time()
        rec = {
            'miou': miou_adv,
            'interrupted': interrupted,
            'ts': t - s,
            'ts_start': str(datetime.fromtimestamp(t)),
            'ts_finish': str(datetime.fromtimestamp(s)),
            'args': vars(args),
        }
        hist.insert(0, rec)

def get_parser() -> ArgumentParser:
    from atk_setting import get_parser as get_base_parser
    parser = get_base_parser()
    parser.add_argument('--limit_img', default=5, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--train_dataset', default='SA1B')
    parser.add_argument('--test_dataset', default='SA1B')
    parser.add_argument('--sta', choices=['train', 'test'], default='test', help='station of comm')
    parser.add_argument('--train_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--test_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--eps', default=10, type=float)
    parser.add_argument('--seed', default=100, type=int, help='rand seed')
    parser.add_argument('--save', default='True', type=bool, help='save the csv')
    parser.add_argument('--train_num', default=100, type=int)
    parser.add_argument('--test_num', default=2000, type=float)
    parser.add_argument('--clean_lim', default=0.0, type=float, help='clean iou lim')
    parser.add_argument('--M', default='vit_b', choices=SAM_CKPTS.keys(), help='model checkpoint')
    return parser

def get_args(parser: ArgumentParser) -> Namespace:
    from atk_setting import get_args as get_base_args
    args = get_base_args(parser)
    args.f = None
    args.D = 'sam'
    args.fps = -1
    args.debug = False
    return args

if __name__ == '__main__':
    parser = get_parser()
    args = get_args(parser)
    device = 'cuda:0'
    if args.test_dataset == 'CITY':
        args.clean_lim = 0.12
    log_save_path = os.path.join('result', 'test', 'log')
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    now_time = make_print_to_file(path=log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    with open(log_save_path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    custom_dataset = choose_dataset(args.test_dataset, args)
    path_id = f"uap_file/{args.test_dataset}.pth"
    test_num, miouimg, miouadv= run(args, path_id, custom_dataset)
    print(f":: miouimg: {miouimg * 100:.2f} %, miouadv: {miouadv * 100:.2f} %")

    if args.save:
        final_log_save_path = os.path.join('result', 'test')
        if not os.path.exists(final_log_save_path):
            os.makedirs(final_log_save_path)
        final_result = []
        final_result_ = {"ckpt": args.M,
                         "seed": args.seed,
                         "now_time": now_time,
                         "final_log_save_path": final_log_save_path,
                         "train_dataset": args.train_dataset,
                         "test_dataset": args.test_dataset,
                         "train_prompt": args.train_prompts,
                         "test_prompt": args.test_prompts,
                         "eps": args.eps,
                         "train_num": args.train_num,
                         "test_num": test_num,
                         "miouimg": f"{miouimg * 100:.2f} %",
                         "miouadv": f"{miouadv * 100:.2f} %"}
        final_result.append(final_result_)
        header = ["ckpt", "seed", "now_time", "final_log_save_path", "train_dataset","test_dataset", "train_prompt", "test_prompt", "eps", "train_num", "test_num", "miouimg", "miouadv"]
        with open(final_log_save_path + f'/final_results.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(final_result)

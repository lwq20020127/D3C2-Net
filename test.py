import torch, os, glob, cv2, random
import numpy as np
from argparse import ArgumentParser
from model import D3C2Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time

parser = ArgumentParser(description='D3C2-Net+')
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--phase_num', type=int, default=25)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--data_dir', type=str, default='./data/test')
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--testset_name', type=str, default='Set11')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--num_feature', type=int, default=32)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--num_rb', type=int, default=2)

args = parser.parse_args()
epoch= args.epoch
T = args.phase_num
B = args.block_size
nf = args.num_feature
k = args.k
nb = args.num_rb
test_data_dir = args.data_dir
test_set_name = args.testset_name

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fixed seed for reproduction
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cs_ratio_list = [0.01, 0.04, 0.1, 0.3, 0.5]


model = D3C2Net(T, nf, nb, k, B)
model = torch.nn.DataParallel(model).to(device)

para_cnt_Phi = model.module.Phi_weight.numel()
para_cnt_Net = sum(p.numel() for p in model.parameters()) - para_cnt_Phi
print('#Param. of Phi', para_cnt_Phi/1e6, 'M')
print('#Param. of Net', para_cnt_Net/1e6, 'M')


model_dir = './%s/' % (args.model_dir)

# test set info
test_image_paths = glob.glob(os.path.join(test_data_dir, test_set_name) + '/*')
test_image_num = len(test_image_paths)


def test(cs_ratio):
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i in range(test_image_num):
            test_image = cv2.imread(test_image_paths[i], 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0])
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
            x = torch.from_numpy(img_pad).float().to(device)
            x_out = model(x, torch.tensor([cs_ratio], device=device))
            x_out = x_out[0,0,:old_h,:old_w].clamp(min=0.0, max=1.0).cpu().numpy() * 255.0
            PSNR = psnr(x_out, img)
            SSIM = ssim(x_out, img, data_range=255)
            # print('[%d/%d] %s, PSNR: %.2f, SSIM: %.4f' % (i, test_image_num, image_path, PSNR, SSIM))
            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)
    return np.mean(PSNR_list), np.mean(SSIM_list)

model.load_state_dict(torch.load('./%s/net_params.pkl' % (model_dir)))

for cs_ratio in cs_ratio_list:
    cur_psnr, cur_ssim = test(cs_ratio)
    log_data = 'CS Ratio is %.2f, PSNR is %.2f, SSIM is %.4f.' % (cs_ratio, cur_psnr, cur_ssim)
    print(log_data)


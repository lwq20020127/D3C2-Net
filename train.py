import torch, os, glob, cv2, random
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from argparse import ArgumentParser
from model import D3C2Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
world_size = dist.get_world_size()

parser = ArgumentParser(description='D3C2-Net+')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int, default=1000)
parser.add_argument('--phase_num', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--block_size', type=int, default=32)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--data_dir', type=str, default='./data/train')
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--save_interval', type=int, default=10)
parser.add_argument('--testset_name', type=str, default='Set11')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--num_feature', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--num_rb', type=int, default=2)

args = parser.parse_args()
start_epoch, end_epoch = args.start_epoch, args.end_epoch
learning_rate = args.learning_rate
batch_size = args.batch_size // world_size
T = args.phase_num
B = args.block_size
nf = args.num_feature
k = args.k
nb = args.num_rb


# fixed seed for reproduction
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

patch_size = 128
iter_num = 1000 * world_size
N = B * B
cs_ratio_list = [0.01, 0.04, 0.1, 0.3, 0.5]

# training set info
print('reading files...')
start_time = time()
training_image_paths = glob.glob(os.path.join(args.data_dir) + '/*')
# training_image_paths += glob.glob(os.path.join(args.data_dir, 'BSD400') + '/*')
# training_image_paths += glob.glob(os.path.join(args.data_dir, 'DIV2K_train_HR') + '/*')
random.shuffle(training_image_paths)
total_images = len(training_image_paths)
per_proc_images = total_images // world_size
start = rank * per_proc_images
end =  start + per_proc_images if rank != world_size - 1 else total_images
training_image_paths = training_image_paths[start:end]


print('training_image_num', len(training_image_paths), 'read time', time() - start_time)

model = D3C2Net(T, nf, nb, k, B)
device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[rank])

para_cnt_Phi = model.module.Phi_weight.numel()
para_cnt_Net = sum(p.numel() for p in model.parameters()) - para_cnt_Phi
print('#Param. of Phi', para_cnt_Phi/1e6, 'M')
print('#Param. of Net', para_cnt_Net/1e6, 'M')

class MyDataset(Dataset):
    def __getitem__(self, index):
        while True:
            path = random.choice(training_image_paths)
            x = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2YCrCb)
            x = torch.from_numpy(x[:, :, 0]) / 255.0
            h, w = x.shape
            max_h, max_w = h - patch_size, w - patch_size
            if max_h < 0 or max_w < 0:
                continue
            start_h = random.randint(0, max_h)
            start_w = random.randint(0, max_w)
            return x[start_h:start_h+patch_size, start_w:start_w+patch_size]

    def __len__(self):
        return iter_num * batch_size

train_dataset = MyDataset()
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=48, pin_memory=True)
optimizer = torch.optim.AdamW([{'params': model.parameters(), 'initial_lr': 1e-4}], lr=learning_rate, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600, 800, 900], gamma=0.1, last_epoch=start_epoch-1)

model_dir = './%s/layer_%d_block_%d_f_%d' % (args.model_dir, T, B, nf)
log_path = './%s/layer_%d_block_%d_f_%d.txt' % (args.log_dir, T, B, nf)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

# test set info
test_image_paths = glob.glob('./data/test/Set11' + '/*')
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

if start_epoch > 0:
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, start_epoch)))

print('start training...')
scaler = GradScaler()
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    print(scheduler.optimizer.param_groups[0]['lr'])

    train_sampler.set_epoch(epoch_i)
    start_time = time()
    loss_avg = 0.0
    dist.barrier()
    for x in tqdm(dataloader):
        x = x.unsqueeze(1).to(device)
        x = H(x, random.randint(0, 7))
        with autocast():
            x_out = model(x, torch.rand(batch_size, device=device))
            loss = ((x_out - x).pow(2) + 1e-4).pow(0.5).mean()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_avg += loss.item()
    scheduler.step()
    loss_avg /= (iter_num // world_size)
    log_data = '[%d/%d] Average loss: %f, time cost: %.2fs.' % (epoch_i, end_epoch, loss_avg, time() - start_time)
    if rank == 0:
        print(log_data)
    with open(log_path, 'a') as log_file:
        if rank == 0:
            log_file.write(log_data + '\n')
    if epoch_i % args.save_interval == 0 and rank == 0:
        torch.save(model.state_dict(), './%s/net_params_%d.pkl' % (model_dir, epoch_i))  # save only the parameters
    if epoch_i == 1 or epoch_i % 2 == 0:
        if rank == 0:
            for cs_ratio in cs_ratio_list:
                cur_psnr, cur_ssim = test(cs_ratio)
                log_data = 'CS Ratio is %.2f, PSNR is %.2f, SSIM is %.4f.' % (cs_ratio, cur_psnr, cur_ssim)
                print(log_data)
                with open(log_path, 'a') as log_file:
                    log_file.write(log_data + '\n')
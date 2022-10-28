from fid import *
import os
import cleanfid
import argparse
import torch
here_dir = '..'
sys.path.append(os.path.join(here_dir, 'src'))

from models import load_model

parser = argparse.ArgumentParser(description="FID")

parser.add_argument("path", type=str, help="path to model checkpoint")
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument(
    "--num_gen", type=int, default=50_000, help="number of fake image used to comute fid"
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="batch size"
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default='bdd_rect_256',
    help="dataset name used when computing real imgs stats with setup_fid.py",
)

args = parser.parse_args()
torch.cuda.set_device(args.device)
print(f'loading model from {args.path}')
model = load_model(args.path, args.device)
def gen_fn(z):
    return model.gen(z, ema=True, norm_img=True)

with torch.no_grad():
    fid_score = cleanfid.fid.compute_fid(gen=gen_fn, dataset_name=args.dataset_name,
                                    dataset_res=256, num_gen=args.num_gen,
                                    dataset_split="custom", device=args.device,
                                    num_workers=8, z_dim=512, batch_size=args.batch_size,
                                    verbose=True)
print('fid score: ', fid_score)
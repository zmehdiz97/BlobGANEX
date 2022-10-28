#from __future__ import absolute_import
import argparse
import os, sys
import torch
from tqdm import tqdm
import numpy as np
from torchvision.transforms import functional as F
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.fid import frechet_distance

here_dir = '..'
sys.path.append(os.path.join(here_dir, 'src'))

from models import load_model



if __name__ == "__main__":
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

    l_feats = []
    model_feat = build_feature_extractor('clean', args.device)

    ref_mu, ref_sigma = get_reference_statistics(args.dataset_name, 256,
                                                    mode='clean', seed=0, split='custom')
    with torch.no_grad():
        for _ in tqdm(range(args.num_gen//args.batch_size)):
            z = torch.randn((args.batch_size, 512)).to(args.device)
            img_batch = model.gen(z, ema=True, norm_img=True, no_jitter=True)
            img_batch = F.resize(img_batch, (299,299)).clip(0, 255)
            feat = model_feat(img_batch).detach().cpu().numpy()
            l_feats.append(feat)
        np_feats = np.concatenate(l_feats)

        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)

        print('fid score: ', fid)
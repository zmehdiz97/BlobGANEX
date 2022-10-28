#from __future__ import absolute_import
import os, sys
import numpy as np
import random
import torch
from tqdm import tqdm
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import torchvision.transforms as T

import lpips
import random
import ipdb

here_dir = '.'

sys.path.append(os.path.join(here_dir, 'src'))

from models import DecisionDensenetModel
from models.networks import StyleGANDiscriminator

from models import load_model
from utils import get_tensor_value, opt_var
from data.utils import CustomImageDataset


class Config:
    blobgan_weights = 'checkpoints/blobgan_256x512.ckpt'
    encoder_weights = 'checkpoints/encoder_256x512.pt'
    decision_model_weights = 'checkpoints/decision_densenet.tar'

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    #decision_model_weights = '/root/workspace/STEEX/checkpoints/decision_model_bdd/checkpoint_biased2.tar'
    output_dir = 'experiments/target_style'
    real_images = True
    bs = 6
    dataset_path = '/datasets_local/BDD/bdd100k/seg/images/val'
    num_imgs=200

    load_inversion=True
    inversions_path = 'shared_files/validation_rec.pt'

class BlobEX():
    def __init__(self, opt: Config):
        self.opt = opt
        os.makedirs(opt.output_dir, exist_ok=True)
        self.model = load_model(opt.blobgan_weights, opt.device)
        self.model.render_kwargs['ret_layout'] = False
        self.model.render_kwargs['norm_img'] = False
        self.model.get_mean_latent()

        self.aspect_ratio = self.model.generator_ema.aspect_ratio
        self.resolution = self.model.resolution

        # Encoder
        self.encoder = StyleGANDiscriminator(size = self.resolution,
                                            discriminate_stddev = False,
                                            d_out=self.model.layout_net_ema.mlp[-1].weight.shape[1],
                                            aspect_ratio = self.aspect_ratio).to(opt.device)  

        self.encoder.load_state_dict(torch.load(opt.encoder_weights)['model']) 
        self.encoder.eval()

        #decision model
        self.decision_model = DecisionDensenetModel(num_classes=4)
        self.decision_model.load_state_dict(torch.load(opt.decision_model_weights)['model_state_dict'])
        self.decision_model.eval().to(opt.device)

        #LPIPS
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(opt.device)
    
        #Dataloader
        if opt.real_images:
            if not(opt.load_inversion):
                self.transform = self.get_transform()
                self.get_dataloader(opt.dataset_path, opt.bs)
                self.num_imgs = len(self.dataset)
            else:
                self.loaded_inversions = torch.load(opt.inversions_path)
                self.num_imgs = len(self.loaded_inversions)
        else:
            self.num_imgs = opt.num_imgs

    def get_transform(self):
        stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}

        if self.aspect_ratio != 1 and type(self.resolution) == int:
            self.resolution = (self.resolution, int(self.aspect_ratio*self.resolution))
        transform = T.Compose([
            t for t in [
                T.Resize(self.resolution, T.InterpolationMode.LANCZOS),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize(stats['mean'], stats['std'], inplace=True),
            ]
        ])
        return transform
    def get_dataloader(self, path, batch_size, shuffle=False):
        self.dataset = CustomImageDataset(path, self.transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, shuffle = shuffle)

    def lpips_loss(self, x1,x2):
        return self.loss_fn_vgg(x1, x2).squeeze()

    def generate_layout_feat_(self, z, truncate=None, mlp_idx=None):
        num_features = random.randint(self.model.n_features_min, self.model.n_features_max)
        if truncate is not None:
            mlp_idx = -1
            z = self.model.layout_net_ema.mlp[:mlp_idx](z)
            z = (self.model.mean_latent * truncate) + (z * (1 - truncate))
        return self.model.layout_net_ema(z, num_features, mlp_idx)

    def invert_and_cf(self):
        opt = self.opt

        num_batches = self.num_imgs // opt.bs
        num_batches += self.num_imgs % opt.bs
        if opt.real_images and not(opt.load_inversion):
                iterator = iter(self.dataloader)
                
        for idx in range(num_batches):
            metadata={}
            
            if opt.real_images:
                if not(opt.load_inversion):
                    batch = next(iterator)
                    # loading target image
                    metadata['image_names'] = batch[1]
                    target = batch[0].to(opt.device)

                    # Encoding image into intermediate latent space(1024) 
                    latent_enc = self.encoder(target).detach()
                    # intermediate latent --> blob params & features
                    blob_enc = self.generate_layout_feat_(latent_enc, mlp_idx=-1)

                    # Optimizing blob parameters for inversion
                    blob_optim, images, lpips_ = self.inv_optim(target, blob_enc)
                    metadata['blob_optim'] = blob_optim
                    metadata['lpips'] = lpips_
                else:
                    metadata['image_names'] = list(self.loaded_inversions.keys())[opt.bs*idx:opt.bs*(idx+1)]
                    aux = list(self.loaded_inversions.values())[opt.bs*idx:opt.bs*(idx+1)]
                    metadata['blob_optim'] = {k: torch.cat([aux[i][k] for i in range(opt.bs)]) for k in aux[0].keys()}
                    target = []
                    for i in range(opt.bs):
                        img_path = os.path.join(opt.dataset_path, metadata['image_names'][i])
                        image = Image.open(img_path)
                        transform = self.get_transform()
                        target.append(transform(image)[None,...])
                    target = torch.cat(target)
                    reconstr_images = self.model.gen(layout=metadata['blob_optim'], **self.model.render_kwargs)

                    images = {
                        'real': target,
                        'reconstr': reconstr_images,
                    }

            else:
                z = torch.randn((opt.bs, 512)).to(opt.device)
                blob_optim = self.generate_layout_feat_(z, truncate=0.3)
                metadata = {}
                with torch.no_grad():
                    imgs = self.model.gen(layout=blob_optim, **self.model.render_kwargs)
                    images = {'query': imgs}    #bs,1,3,256,256
                metadata['blob_optim'] = blob_optim #{ k:v.cpu() for k, v in blob_optim.items()}
            
            # Optimizing blob parameters for cf
            metadata, images = self.cf_optim(metadata, images, target.to(opt.device))
            self.save_results(metadata, images, idx)

    def save_results(self, metadata, images, idx):
        images = {k: v.float().cpu() for k, v in images.items()}
        images = torch.cat([v for v in images.values()], 0)
        image_grid = make_grid(
            images, normalize=True, value_range=(-1, 1), nrow=self.opt.bs
        )
        image_grid = F.to_pil_image(image_grid)
        image_grid = image_grid.save(f"{self.opt.output_dir}/snapshot_{idx}.jpg")
        torch.save(metadata, f'{self.opt.output_dir}/metadata_{idx}.pt')
    
    def inv_optim(self, im_tar, blob_enc):

        ## hyper parameters
        learning_rate = 0.02
        n_iters = 400
        target_features = ['xs', 'ys', 'sizes', 'covs', 'features', 'spatial_style']
        lr = {'xs':learning_rate, 'ys':learning_rate, 'sizes':learning_rate*2, 'covs':learning_rate} #/5 /5 /1 /10

        with torch.no_grad():
            img_init = self.model.gen(layout=blob_enc, **self.model.render_kwargs)
            target_decision_feat = self.decision_model.feat_extract(im_tar)

        blob_optim = opt_var(blob_enc, target_params=target_features)

        params = []
        for key, val in blob_optim.items():
            if key in target_features:
                params.append({'params':val, 'lr':lr.get(key, learning_rate)})

        optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        viz_tar = im_tar.clone().detach()
        viz_init = img_init.clone().detach()

        pbar = tqdm(range(1, n_iters + 1), leave=True)#, desc=f'Image {i}')
        prox_criterion = torch.nn.MSELoss()
        for step in pbar:
            img =  self.model.gen(layout=blob_optim, **self.model.render_kwargs)
            
            loss_l2 = torch.mean((img - im_tar) ** 2, dim=(1,2,3))
            log_message = f'loss_l2: {get_tensor_value(loss_l2).mean():.4f}'
            
            loss_pips = self.lpips_loss(img, im_tar)
            log_message += f', LPIPS: {get_tensor_value(loss_pips).mean():.4f}'
            
            decision_feat = self.decision_model.feat_extract(img)
            decision_loss = torch.mean((decision_feat - target_decision_feat) ** 2)
            log_message += f', Decision: {get_tensor_value(decision_loss).mean():.4f}'
            
            loss_prox = 0
            for (k_opt, v_opt), (k_orig, v_orig) in zip(blob_optim.items(), blob_enc.items()):
                loss_prox +=  prox_criterion(v_opt,v_orig)
            log_message += f', L_prox: {get_tensor_value(loss_prox).mean():.4f}'
            
            loss = loss_pips + 0.1*loss_l2 + 0.1*decision_loss + 0.1*loss_prox

            pbar.set_description_str(log_message)
            loss = loss.sum()
            # Do optimization.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        viz_img = img.clone().detach()
        imgs = {
            'real': viz_tar,
            'enc': viz_init,
            'reconstr': viz_img,
        }
        return opt_var(blob_optim), imgs, loss_pips.detach().cpu()

    def cf_optim(self, metadata, images, im_tar):

        # Hyperparameters
        target_attributes = [0,1,2,3] #
        target_features = ['spatial_style']
        learning_rate = 0.08
        n_iters = 100
        lr = {'xs':learning_rate/8, 'ys':learning_rate/8, 'sizes':learning_rate,
        'covs':learning_rate/8, 'features':learning_rate/8, 'spatial_style':learning_rate/8} #/5 /5 /1 /10
        λ_prox = 0 #1 5 0 0.1 10
        λs = {'spatial_style':1}

        criterion = torch.nn.L1Loss()
        layout_metadata_orig = metadata['blob_optim']
        scale_features = torch.linalg.norm(layout_metadata_orig['features'], dim=-1, keepdim=True)
        scale_spatial_style = torch.linalg.norm(layout_metadata_orig['spatial_style'], dim=-1, keepdim=True)

        for target_attribute in target_attributes:
            with torch.no_grad():
                img_orig = self.model.gen(layout=layout_metadata_orig, **self.model.render_kwargs)
                initial_scores = self.decision_model(img_orig)
                metadata['init_scores'] = initial_scores.detach().cpu()
            # target = 0 if initial_score > 0.5, else target = 1
            target = (initial_scores[:, target_attribute] < 0.5).double()

            layout_metadat_opt = opt_var(layout_metadata_orig, target_params=target_features)
            # defining params and lr
            params = []
            for key, val in layout_metadat_opt.items():
                if key in target_features:
                    params.append({'params':val, 'lr':lr[key]})
            optimizer = torch.optim.Adam(params, lr=learning_rate)
            pbar = tqdm(range(1, n_iters + 1), leave=True)#, desc=f'Image {i}')
            for step in pbar:
                log_message = f'Att {target_attribute} || '
                norm_features = torch.linalg.norm(layout_metadat_opt['features'], dim=-1, keepdim=True).detach()
                norm_spatial_style = torch.linalg.norm(layout_metadat_opt['spatial_style'], dim=-1, keepdim=True).detach()
                layout_metadat_opt_scaled = {}
                layout_metadat_opt_scaled['features'] = scale_features * layout_metadat_opt['features'] / norm_features
                layout_metadat_opt_scaled['spatial_style'] = scale_spatial_style * layout_metadat_opt['spatial_style'] / norm_spatial_style
                layout_metadat_opt_scaled = {**layout_metadat_opt_scaled, **layout_metadat_opt}
                img =  self.model.gen(layout=layout_metadat_opt, **self.model.render_kwargs)
                counterfactual_probas = self.decision_model(img)

                # Decision loss
                flip_decision_loss = - (1 - target) * torch.log(1 - counterfactual_probas[:, target_attribute]) \
                    - target * torch.log(counterfactual_probas[:, target_attribute])
                log_message += f'L_decision: {get_tensor_value(flip_decision_loss).mean():.4f}'
                
                loss_prox = 0
                for (k_opt, v_opt), (k_orig, v_orig) in zip(layout_metadat_opt.items(), layout_metadata_orig.items()):
                    λ = λs.get(k_opt, 0)
                    loss_prox += λ * criterion(v_opt,v_orig)
                log_message += f', L_prox: {get_tensor_value(loss_prox).mean():.4f}'

                pbar.set_description_str(log_message)
                loss = (flip_decision_loss + λ_prox * loss_prox).sum()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(lambda: loss)

            metadata[f'att_{target_attribute}'] = {
                'final_scores':counterfactual_probas.detach().cpu(),
                'blob_cf': {k: v.detach().cpu() for k, v in layout_metadat_opt.items()},
                'lpips': self.lpips_loss(img, im_tar).detach().cpu()
            }

            images[f'CF_{target_attribute}'] = img
        return metadata, images

if __name__ == "__main__":
    opt = Config()
    blobex = BlobEX(opt)
    blobex.invert_and_cf()


# Training blobgan
use config files stored in src/configs, more specifically the blobgan40.yaml to train the blobgan
It contains all useful parameters for training (number of GPUs, batchsize, checkpointing, logging, ..)
- logging:
  - Put your own wandb account name in wandb.entity. If you don't want to log in to wandb use wandb.offline=true
  - command to run training: WANDB_API_KEY=<API KEY> python src/run.py +experiment=blobgan40
  - files will be stored in logs/wandb (you may need to mkdir logs/wandb if it doesn't exist already)
  - Each new training will be sored in a separate folder inside logs/wandb under the name "run-DATE_TIME_id" 
  - You can resume training with resume.id (eg. resume.id=36lxjdji) otherwise resume.id=null
- FID monitoring
  - Avoid computing FID during training (it slows down the training: ~1h for each fid computation). If you still want to set these params in the yaml config file:
  checkpoint:
    every_n_train_steps: 5000
    save_top_k: -1 # (-1 --> keeps all past checkpoints) (5 --> keeps last 5 checkpoints)
    mode: max
    monitor: train/fid (monitor: step if you don't want to monitor the fid)
  - You should compute the stats of the training dataset beforehand using setup_fid.py (see fid section). the setup_fid.py will precompute stats over real images and saves them in a file (with a given name). Then you need to specify the name of stats when running the training (eg. model.fid_stats_name='bdd_rect_256')
   
Practical info:
- Training requires large GPUs (urus20/viper/urus40)
- max batchsize possible between 4 and 12 (try to use trainer.accumulate_grad_batches if the batchsize doesn't seem enough)
- model.generator.size_in is the size of the input grid from which we start the generation (model.generator.size_in=8 and model.aspect_ratio=2 ==> input_size=(8,16))
- To change size of feature vectors (style and spatial) use generator.override_c_in (size of spatial feature vector) and layoutnet.feature_dim (size of style vector)

# FID
- Don't use uppercase characters in the name (the name will be transformed to lowercase)
- Command:
  python scripts/setup_fid.py --action compute_new --n_imgs 50_000 --shuffle --name bdd_rectangle_256 -bs 32 --device cuda --path /datasets_local/BDD/bdd100k/images/100k/train
- This computes stats of real images and stores them for later use. Takes 7-12mins on 50_000 images.
- Then, use either compute_fast_fid.py or compute_clean_fid.py to compute the fid score of a gan.
  python compute_fast_fid.py checkpoints/blobgan_256x512.ckpt --num_gen 50_000 --batch_size 16 --dataset_name bdd_rect_256
- Same command with compute_clean_fid.py (more precise fid using PIL resizing functions (CVPR'22))

PS: setup_fid.py will save satat results inside /opt/conda/lib/python3.10/site-packages/cleanfid which is not practical because you will need to run the setup_fid.py at each new job. You can save it locally and add commands in the docker file to save it at that location. Or use stats saves in the fid_stats/ directory.

# training the encoder
1) Training on generated images
python pretrain_encoder.py 
Arguments and parameters are hardcoded at the beginning of the script(lr, bs, number of iterations, path to blobgan weights and mlp_idx). mlp_idx is the target layer of the layoutnetwork where we want to do the inverion. eg. mlp_idx=-1 (which I used) means that we are using the latent space before the last layer of the layoutnetwork which has a reasonable size of 1024. Inverting directly in the blob space requires a very large encoder {blobspace size = #blobs * (style_size + spatial_size + spatial_params) ≈ 40 *(256+256) = 20480}

2) Training on both generated and real images
train_encoder.py

Comments on the encoder inversion: 
- Training time 24h-48h (both trainings)
- The performance of the inversion is a real bottleneck (especially the encoder part)
- I didn't spend a lot of time finetuning the hyperparams. The performance of the encoder is still poor. - The decision loss I added doesn't seem to have an impact on the performance (however it has some effect on the inversion with optimization). 
- Some ideas that I have tried: changing the architecture (currently I am using the stylegan discriminator architecture for the encoder). I tried a resnet architecture which gave slightly better results on square images but lower performance on rectangular images.
- There is a big difference between the performance of the encoder on square images and rectangular ones (especially in recovering cars)

# inversion and cf
1) basic cf generation:
python blobex_inv_cf.py
- Both inversion and cf generation loops are implemented to optimize batches of images 

- The Config class contains the main parameters to run an experiment: paths to Blobgan and decision model checkpoints, batch size, using real images or real ones -->(inverting first).
- The optimization parameters are defined inside the inv_optim and cf_optim methods (learning rate, lambdas, target_attributes, number of iterations, parameters to optimize (x,a,s,phi, ..))
eg.
target_attributes = [0,1,2,3] # Forward, stop, left, right
target_features = ['xs', 'ys', 'sizes', 'covs', 'features', 'spatial_style'] # covs if the rotation 

!important 
the background has only 'features' and 'spatial_style', so the shape of the tensors will be different (see demo.ipynb)
['features', 'spatial_style'] : K+1 (background features are stored in the first value)
['xs', 'ys', 'covs']: K
's': K+1 (but the first value is never used in the image generation)

Comments on hyperparameters:
- Hyperparameters are difficult to finetune. They are also highly correlated (num_iter <--> learning rate <--> lambda_prox). 
- They also depend on the configuration: square or rectangular images
- For cf hyperparameters, change the weights of the proximity loss depending on the amount and type of change you want. 
- lr and n_iters should be changed together (eg. if you decrease lr a lot, increase the number of iterations). 
- As you can see in the code, the learning rate is higher for the size parameter --> so that it becomes easier to add objects in the scene (because the gradient flowing through inactive blobs is small). For visualization it is better, but it will probably yield a higher LPIPS (didn't investigate this claim).

2) Blob targeting
python blobex_inv_cf_target_blobs.py
The same code is written in blobex_inv_cf.py with a slight modification to set the requires_grad to a subset of blobs only. 

Use the 'target_blob' in the cf_optim method to specify a list of blob indices

3) Generating simultaneously diverse CFs: 
python blobex_inv_cf_diversity.py

same code with other changes in the cf_optim method (new diversity loss, Assignment matrix)

The 3 above scripts work for generated images and real images 
The scripts will save two types of files:
- snapshots (one per batch): image grid with all images of the batch, reconstructions, and CFs
- a metadat.pt file (one per batch): contains all information needed to run further experiments 
'image_names'
'latent_enc' (removed but present in some old metadata files)
'blob_optim' (result of inv optim)
'init_scores' (initial pred of the decision model)
'att_0' 
    'final_scores' (final scores)
    'blob_cf' (reult if cf optim)

'att_1', 'att_2', 'att_3'

# Checkpoints
Generator 256x256: checkpoints/blobgan_256x256.ckpt
Generator 256x512: checkpoints/blobgan_256x512.ckpt
Encoder 256x256: checkpoints/encoder_256x256.ckpt
Encoder 256x512: checkpoints/encoder_256x512.ckpt
Decision model: checkpoints/decision_densenet.tar
Decision model biased: checkpoints/decision_densenet_biased.tar
Object detection: checkpoints/detection
Semantic segmentation: checkpoints/sem_seg
Panoptic segmentation: checkpoints/panoptic
Drivable area segmentation: checkpoints/drivable


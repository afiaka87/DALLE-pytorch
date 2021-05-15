import torch
from dalle_pytorch import CLIP
import argparse
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dalle_pytorch import CLIP
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer

# quit early if you used the wrong folder name

assert Path('/mnt/samsung_t7_beta/DataImportant').exists(), f'The path  was not found.'

# helpers

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir

# constants

EPOCHS = 20
BATCH_SIZE = 1

LEARNING_RATE = 1e-3
is_shuffle = False

TEXT_SEQ_LEN = 64

ds = TextImageDataset(
    '/mnt/samsung_t7_beta/DataImportant',
    text_len=TEXT_SEQ_LEN,
    image_size=128,
    resize_ratio=0.75,
    truncate_captions=True,
    tokenizer=tokenizer,
    shuffle=is_shuffle,
)

assert len(ds) > 0, 'dataset is empty'

data_sampler = None

dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=data_sampler)

# initialize DALL-E



clip = CLIP(
    dim_text = 128,
    dim_image = 128,
    dim_latent = 128,
    num_text_tokens = 10000,
    text_enc_depth = 4,
    text_seq_len = 128,
    text_heads = 1,
    num_visual_tokens = 512,
    visual_enc_depth = 4,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 1
)

clip = clip.cuda()


opt = Adam(clip.parameters(), lr=1e-3)

# training

for epoch in range(EPOCHS):
    for i, (text, images) in enumerate(dl):
        text, images = map(lambda t: t.cuda(), (text, images))

        mask = torch.ones_like(text).bool()
        loss = clip(text, images, text_mask = mask, return_loss = True)
        loss.backward()

        loss.backward()
        clip_grad_norm_(clip.parameters(), GRAD_CLIP_NORM)
        opt.step()
        opt.zero_grad()


        log = {}

        if i % 10 == 0:
            print(epoch, i, f'loss - {loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item()
            }


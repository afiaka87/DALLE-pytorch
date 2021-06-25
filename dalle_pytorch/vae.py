import os
import urllib
from math import log, sqrt
from pathlib import Path
from symbol import import_as_name

import PIL
import yaml
from einops import rearrange
from omegaconf import OmegaConf
from taming.models.vqgan import GumbelVQ, VQModel
from tqdm import tqdm

from dalle_pytorch import distributed_utils


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = PIL.Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def reconstruct_with_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec


# constants

CACHE_PATH = os.path.expanduser("~/.cache/dalle")

OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'

VQGAN_VAE_PATH = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
VQGAN_VAE_CONFIG_PATH = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'

# helpers methods


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))


def map_pixels(x, eps=0.1):
    return (1 - 2 * eps) * x + eps


def unmap_pixels(x, eps=0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)


def download(url, filename=None, root=CACHE_PATH):
    if (not distributed_utils.is_distributed
            or distributed_utils.backend.is_local_root_worker()):
        os.makedirs(root, exist_ok=True)
    filename = default(filename, os.path.basename(url))

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if (distributed_utils.is_distributed
            and not distributed_utils.backend.is_local_root_worker()
            and not os.path.isfile(download_target)):
        # If the file doesn't exist yet, wait until it's downloaded by the root worker.
        distributed_utils.backend.local_barrier()

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target_tmp,
                                                     "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")),
                  ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)
    if (distributed_utils.is_distributed
            and distributed_utils.backend.is_local_root_worker()):
        distributed_utils.backend.local_barrier()
    return download_target


# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841


class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None, is_gumbel=True):
        model_filename = 'vqgan_gumbel.8192.model.ckpt'
        config_filename = 'vqgan_gumbel.8192.config.yaml'
        config_path = str(Path(CACHE_PATH) / config_filename)
        model_path = str(Path(CACHE_PATH) / model_filename)
        if vqgan_model_path is None:
            download(VQGAN_VAE_CONFIG_PATH, config_filename)
            download(VQGAN_VAE_PATH, model_filename)
        else:
            model_path = vqgan_model_path
            config_path = vqgan_config_path
        if is_gumbel:
            model = GumbelVQ(**config.model.params)
        else:
            model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Loaded VQGAN from {model_path} and {config_path}")
        
        self.model = model

        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        self.num_layers = int(log(f) / log(2))
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(
            self, self.model.quantize.embedding.weight)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices, '(b n) () -> b n', b=b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq,
                                    num_classes=self.num_tokens).float()
        z = (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h=int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented
# pretrained Discrete VAE from OpenAI


class OpenAIDiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = load_model(download(OPENAI_VAE_ENCODER_PATH))
        self.dec = load_model(download(OPENAI_VAE_DECODER_PATH))

        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc.blocks(img)
        z = torch.argmax(z_logits, dim=1)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h=int(sqrt(n)))

        z = F.one_hot(img_seq, num_classes=self.num_tokens)
        z = rearrange(z, 'b h w c -> b c h w').float()
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        raise NotImplemented


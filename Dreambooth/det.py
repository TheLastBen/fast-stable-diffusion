#Adapted from A1111
import argparse
import torch
import open_clip
import transformers.utils.hub
from safetensors import safe_open
import os
import sys
import wget
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("--MODEL_PATH", type=str)
parser.add_argument("--from_safetensors", action='store_true')
args = parser.parse_args()

wget.download("https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/ldm.zip")
call('unzip ldm', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
call('rm ldm.zip', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))

import ldm.modules.diffusionmodules.openaimodel
import ldm.modules.encoders.modules

class DisableInitialization:

    def __init__(self, disable_clip=True):
        self.replaced = []
        self.disable_clip = disable_clip

    def replace(self, obj, field, func):
        original = getattr(obj, field, None)
        if original is None:
            return None

        self.replaced.append((obj, field, original))
        setattr(obj, field, func)

        return original

    def __enter__(self):
        def do_nothing(*args, **kwargs):
            pass

        def create_model_and_transforms_without_pretrained(*args, pretrained=None, **kwargs):
            return self.create_model_and_transforms(*args, pretrained=None, **kwargs)

        def CLIPTextModel_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
            res = self.CLIPTextModel_from_pretrained(None, *model_args, config=pretrained_model_name_or_path, state_dict={}, **kwargs)
            res.name_or_path = pretrained_model_name_or_path
            return res

        def transformers_modeling_utils_load_pretrained_model(*args, **kwargs):
            args = args[0:3] + ('/', ) + args[4:]
            return self.transformers_modeling_utils_load_pretrained_model(*args, **kwargs)

        def transformers_utils_hub_get_file_from_cache(original, url, *args, **kwargs):

            if url == 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/added_tokens.json' or url == 'openai/clip-vit-large-patch14' and args[0] == 'added_tokens.json':
                return None

            try:
                res = original(url, *args, local_files_only=True, **kwargs)
                if res is None:
                    res = original(url, *args, local_files_only=False, **kwargs)
                return res
            except Exception as e:
                return original(url, *args, local_files_only=False, **kwargs)

        def transformers_utils_hub_get_from_cache(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_utils_hub_get_from_cache, url, *args, **kwargs)

        def transformers_tokenization_utils_base_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_tokenization_utils_base_cached_file, url, *args, **kwargs)

        def transformers_configuration_utils_cached_file(url, *args, local_files_only=False, **kwargs):
            return transformers_utils_hub_get_file_from_cache(self.transformers_configuration_utils_cached_file, url, *args, **kwargs)

        self.replace(torch.nn.init, 'kaiming_uniform_', do_nothing)
        self.replace(torch.nn.init, '_no_grad_normal_', do_nothing)
        self.replace(torch.nn.init, '_no_grad_uniform_', do_nothing)

        if self.disable_clip:
            self.create_model_and_transforms = self.replace(open_clip, 'create_model_and_transforms', create_model_and_transforms_without_pretrained)
            self.CLIPTextModel_from_pretrained = self.replace(ldm.modules.encoders.modules.CLIPTextModel, 'from_pretrained', CLIPTextModel_from_pretrained)
            self.transformers_modeling_utils_load_pretrained_model = self.replace(transformers.modeling_utils.PreTrainedModel, '_load_pretrained_model', transformers_modeling_utils_load_pretrained_model)
            self.transformers_tokenization_utils_base_cached_file = self.replace(transformers.tokenization_utils_base, 'cached_file', transformers_tokenization_utils_base_cached_file)
            self.transformers_configuration_utils_cached_file = self.replace(transformers.configuration_utils, 'cached_file', transformers_configuration_utils_cached_file)
            self.transformers_utils_hub_get_from_cache = self.replace(transformers.utils.hub, 'get_from_cache', transformers_utils_hub_get_from_cache)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, field, original in self.replaced:
            setattr(obj, field, original)

        self.replaced.clear()


def vpar(state_dict):

    device = torch.device("cuda")

    with DisableInitialization():
        unet = ldm.modules.diffusionmodules.openaimodel.UNetModel(
            use_checkpoint=True,
            use_fp16=False,
            image_size=32,
            in_channels=4,
            out_channels=4,
            model_channels=320,
            attention_resolutions=[4, 2, 1],
            num_res_blocks=2,
            channel_mult=[1, 2, 4, 4],
            num_head_channels=64,
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=1,
            context_dim=1024,
            legacy=False
        )
        unet.eval()

    with torch.no_grad():
        unet_sd = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items() if "model.diffusion_model." in k}
        unet.load_state_dict(unet_sd, strict=True)
        unet.to(device=device, dtype=torch.float)

        test_cond = torch.ones((1, 2, 1024), device=device) * 0.5
        x_test = torch.ones((1, 4, 8, 8), device=device) * 0.5

        out = (unet(x_test, torch.asarray([999], device=device), context=test_cond) - x_test).mean().item()

    return out < -1


def detect_version(sd):

    sys.stdout = open(os.devnull, 'w')

    sd2_cond_proj_weight = sd.get('cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight', None)
    diffusion_model_input = sd.get('model.diffusion_model.input_blocks.0.0.weight', None)

    if sd2_cond_proj_weight is not None and sd2_cond_proj_weight.shape[1] == 1024:

        if vpar(sd):
          sys.stdout = sys.__stdout__
          sd2_v=print("V2.1-768px")
          return sd2_v
        else:
            sys.stdout = sys.__stdout__
            sd2=print("V2.1-512px")
            return sd2

    else:
      sys.stdout = sys.__stdout__
      v1=print("1.5") 
      return v1


if args.from_safetensors:

    checkpoint = {}
    with safe_open(args.MODEL_PATH, framework="pt", device="cuda") as f:
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)
    state_dict = checkpoint
else:
    checkpoint = torch.load(args.MODEL_PATH, map_location="cuda")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

detect_version(state_dict)

call('rm -r ldm', shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))

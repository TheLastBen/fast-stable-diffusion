import argparse
import os
import sys
import modules.safe

script_path = f'{os.getcwd()}-webui'
models_path = os.path.join(script_path, "models")
sys.path.insert(0, script_path)

# search for directory of stable diffusion in following places
sd_path = None
possible_sd_paths = [os.path.join(script_path, '/content/gdrive/MyDrive/sd/stable-diffusion'), '.', os.path.dirname(script_path)]
for possible_sd_path in possible_sd_paths:
    if os.path.exists(os.path.join(possible_sd_path, 'ldm/models/diffusion/ddpm.py')):
        sd_path = os.path.abspath(possible_sd_path)
        break

assert sd_path is not None, "Couldn't find Stable Diffusion in any of: " + str(possible_sd_paths)

path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion', []),
    (os.path.join(sd_path, 'src/taming-transformers'), 'taming', 'Taming Transformers', []),
    (os.path.join(sd_path, 'src/codeformer'), 'inference_codeformer.py', 'CodeFormer', []),
    (os.path.join(sd_path, 'src/blip'), 'models/blip.py', 'BLIP', []),
    (os.path.join(sd_path, 'src/k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
]

paths = {}

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        if "atstart" in options:
            sys.path.insert(0, d)
        else:
            sys.path.append(d)
        paths[what] = d

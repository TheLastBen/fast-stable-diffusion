# fast-stable-diffusion colabs, +25% speed increase + memory efficient.
2 Colab adaptations for both hlky AUTOMATIC1111 Webui versions of stable diffusion implementing the optimization suggested by https://github.com/MatthieuTPHR : https://github.com/huggingface/diffusers/pull/532, using 
the MemoryEfficientAttention implementation from xformers (cc. @fmassa, @danthe3rd, @blefaudeux) to both speedup the cross-attention speed and decrease its GPU memory requirements.

All you have to do is enter your huggingface token only once and you're all set, the colabs will install the repos and the models inside Gdrive, so the loading will be fast everytime you use it, enjoy !!

hlky WEBUI
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_hlky.ipynb)

AUTOMATIC1111 WEBUI
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb)

Minimalistic relaxed mode if you just want to have fun
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_relaxed.ipynb)

If you encounter any issue or you want to update to latest webui version, remove the folder "sd" or "stable-diffusion-webui" from your GDrive (and GDrive trash) and rerun the colab.

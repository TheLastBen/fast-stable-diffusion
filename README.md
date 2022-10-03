# fast-stable-diffusion colabs, +25-50% speed increase + memory efficient + DreamBooth
2 Colab adaptations for both the HLKY and AUTOMATIC1111 web-ui versions of Stable Diffusion, implementing the optimization suggested by [@MatthieuTPHR](https://github.com/MatthieuTPHR) in [this PR](https://github.com/huggingface/diffusers/pull/532), using 
the MemoryEfficientAttention implementation from xformers (by @fmassa, @danthe3rd, @blefaudeux) to both speed up the cross-attention speed and decrease its GPU memory requirements.

All you have to do is enter your Huggingface token once and you're all set, the colabs will install the repos and the models inside Google Drive, so the loading will be fast everytime you use it. Enjoy!!

hlky WEBUI
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_hlky.ipynb)

AUTOMATIC1111 WEBUI
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb)

Minimalistic relaxed mode if you just want to have fun
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_relaxed.ipynb)

If you encounter any issue or you want to update to latest webui version, remove the folder "sd" or "stable-diffusion-webui" from your GDrive (and GDrive trash) and rerun the colab.

# fast-dreambooth colab, +65% speed increase + less than 14GB VRAM, CKPT output.
Train your model using this easy simple and fast colab, all you have to do is enter you huggingface token once, and it will cache all the files in GDrive, including the trained model and you will be able to use it directly from the colab, make sure you use high quality reference pictures for the training. Thanks to https://github.com/jachiam for his diffusers to CKPT [script](https://gist.github.com/jachiam/8a5c0b607e38fcc585168b90c686eb05), the trained models now can now be used in SD WEBUI from AUTOMATIC1111 and hlky and others.

DreamBooth [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb)

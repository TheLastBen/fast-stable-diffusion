import os
from IPython.display import clear_output
import time

def oldmdl(token):

    if token == "" and not os.path.exists('/content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt'):
      token=input("Insert your huggingface token :")
      clear_output()
      !git init
      !git lfs install --system --skip-repo
      !git clone "https://USER:{token}@huggingface.co/CompVis/stable-diffusion-v1-4"
      if os.path.exists('/content/stable-diffusion-v1-4'):
        !wget https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/convertosd.py
        !sed -i '201s@.*@    model_path = "/content/stable-diffusion-v1-4"@' /content/convertosd.py
        !sed -i '202s@.*@    checkpoint_path= "/content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt"@' /content/convertosd.py
        clear_output()      
        !python /content/convertosd.py 
        if os.path.exists('/content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt'):
          !cp /content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt /content/mainmodel.ckpt
          model='/content/mainmodel.ckpt'        
          clear_output()
          print('[1;32mDONE !')
        else:
          print('[1;31mSomething went wrong, try again')
      else:
        print('[1;31mMake sure you accept the terms at https://huggingface.co/CompVis/stable-diffusion-v1-4')


    elif not os.path.exists('/content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt'):
      clear_output()
      !git init
      !git lfs install --system --skip-repo
      !git clone "https://USER:{token}@huggingface.co/CompVis/stable-diffusion-v1-4"
      if os.path.exists('/content/stable-diffusion-v1-4'):
        !wget https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dreambooth/convertosd.py
        !sed -i '201s@.*@    model_path = "/content/stable-diffusion-v1-4"@' /content/convertosd.py
        !sed -i '202s@.*@    checkpoint_path= "/content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt"@' /content/convertosd.py
        clear_output()       
        !python /content/convertosd.py 
        if os.path.exists('/content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt'):
          !cp /content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt /content/mainmodel.ckpt
          model='/content/mainmodel.ckpt'
          clear_output()
          print('[1;32mDONE !')
        else:
          print('[1;31mSomething went wrong, try again')
      else:
        print('[1;31mMake sure you accept the terms at https://huggingface.co/CompVis/stable-diffusion-v1-4')

    elif not os.path.exists('/content/mainmodel.ckpt') and os.path.exists('/content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt'):
      !cp /content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt /content/mainmodel.ckpt
      model='/content/mainmodel.ckpt'
      clear_output()
      print('[1;32mDONE !')

    elif os.path.exists('/content/mainmodel.ckpt') and os.path.exists('/content/gdrive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion/model.ckpt'):
      model='/content/mainmodel.ckpt'
      clear_output()
      print('[1;32mDONE !')

    if os.path.exists('/content/.git'):
    !rm -r /content/.git

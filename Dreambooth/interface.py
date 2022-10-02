import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from IPython.display import display
import gradio as gr
import os
import random

pipeline = StableDiffusionPipeline.from_pretrained("/content/gdrive/MyDrive/models/instance",torch_dtype=torch.float16).to("cuda")
def dummy(images, **kwargs):
    return images, False
pipeline.safety_checker = dummy


css="""

.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }

[data-testid="image"] {min-height: 512px !important}
* #body>.col:nth-child(2){width:50%;max-width:89vw}
#generate{width: 100%}

#txtset{
    /* min-width: min(320px, 100%); */
    /* flex-grow: 1; */ 
    max-width: 530px !important;
}



input{-webkit-appearance: sliderthumb-horizontal;
    accent-color: darkseagreen;
}

img{
    display: block;
    vertical-align: middle;
    background-color: #1f2937;
}

.dark .dark\:bg-gray-900 {
    --tw-bg-opacity: 1;
    background-color: #1f2937;
}

.border-solid {
    border-style: groove;
}

#prompt_input input,
#prompt_input textarea {
    font-size: 1.2rem;
    line-height: 1.6rem;
}

#prompt_row input,
#prompt_row textarea {
    font-size: 1.2rem;
    line-height: 1.6rem;
}

[data-testid="image"] {
    min-height: 751px !important;

}

#img2img_mask_btn, #img2img_edit_btn {
    align-self: stretch;
    max-width: 50px;
    max-height: 69px;
}
 
#prompt_row{height:150%}

.gr-text-input:disabled {
    --tw-shadow: 0 0 #0000;
    --tw-shadow-colored: 0 0 #0000;
    box-shadow: var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow);
    color: darkgray;
}

.xl\:min-h-\[450px\] {
    min-height: 754px !important;
  }
  
::before,
::after {
  box-sizing: border-box; /* 1 */
  border-width: 0; /* 2 */
  border-style: none; /* 2 */
  border-color: #e5e7eb; /* 2 */
}

@media (min-width: 1536px)
.\32xl\:max-h-\[20rem\] {
    max-height: 40rem;
}

.gallery-item.svelte-1g9btlg.svelte-1g9btlg{position:relative;aspect-ratio:1 / 0 !important;
height:89%;
width:78%;
overflow:hidden;
border-radius:0.50rem !important;
--tw-bg-opacity:1;
background-color:rgb(243 244 246 / var(--tw-bg-opacity));
object-fit:fill;
--tw-shadow:0 1px 2px 0 rgb(0 0 0 / 0.05);
--tw-shadow-colored:0 1px 2px 0 var(--tw-shadow-color);
box-shadow:var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow);
outline:0px solid transparent;outline-offset:0px !important;
--tw-ring-offset-shadow:var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
--tw-ring-shadow:var(--tw-ring-inset) 0 0 0 calc(1px + var(--tw-ring-offset-width)) var(--tw-ring-color);
box-shadow:var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
--tw-ring-opacity:1;
--tw-ring-color:rgb(229 231 235 / var(--tw-ring-opacity))

}

.grid {
    display: grid;
    justify-items: center;
}

.h-\[calc\(100\%-50px\)\] {
    height: calc(96% - 50px);
}

.pt-6 {
    padding-top: 2.5rem;
}

.my-4 {
    margin-top: -0.8rem;
    margin-bottom: 1rem;
}

.gr-button-lg {
    border-radius: 0.5rem;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
    font-size: 1rem;
    line-height: 1.5rem;
    font-weight: 60;
	
}



.container {
    max-width: 1903px;
}

* #body>.col:nth-child(2) {
    width: 35%;
    max-width: 89vw;
}

"""

def sd(x):
  z=torch.Generator(device='cuda').manual_seed(x)
  return z

with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
    with gr.Tabs(elem_id='tabss') as tabs:
        with gr.TabItem("Stable Diffusion Text to Image", id='txt2img_tab'):

            with gr.Row(elem_id='body').style(equal_height=False):
                with gr.Column():
                    with gr.Group(elem_id="prompt_row"):
                        output_txt2img_gallery = gr.Gallery(label="Images", elem_id="txt2img_gallery_output").style(grid=[2,3], container=True)
                        prompt = gr.Textbox(label="Prompt", placeholder="Prompt", elem_id='prompt_input', lines=3, max_lines=4, value='', show_label=False).style()
                        txt2img_btn = gr.Button("Generate", elem_id="generate", variant="primary")
                   
                with gr.Column(elem_id="txtset"):
                    alphas=gr.Textbox(label='Seed',interactive=True, visible=False, placeholder='')
                    width = gr.Slider(minimum=512, maximum=2048, step=64, label="Width", value=512)
                    height = gr.Slider(minimum=512, maximum=2048, step=64, label="Height", value=512)
                    guidance_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale', value=7)
                    num_inference_steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=35)
                    batch_size = gr.Slider(minimum=1, maximum=25, step=1, label='Number of Images', value=1)
                    #txt2img_sampling = gr.Dropdown(label='Sampling Method', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value='k_euler_a')
                     
                                

                     #           txt2img_seed_type=gr.Checkbox(label="Random Seed", value=True)                        
                    #            seed_btn = gr.Button("Keep Current Seed")
                           #     txt2img_variant_amount = gr.Slider(minimum=0.0, maximum=1.0, label='Variation Amount',value=0.0, interactive=True)


				

          
                   #with gr.Row():                           
                        #output_txt2img_copy_to_input_btn = gr.Button("Send to img2img")
                        #output_txt2img_copy_to_uncrop_btn = gr.Button("Send to Enhancements")
        def t2i(prompt, height, width, num_inference_steps, guidance_scale, batch_size):

           path='/content/gdrive/MyDrive/Output_images/'
           
           
           with autocast("cuda"):
               images = pipeline([prompt]*batch_size, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images                 
               for k in images:
                  name=(prompt[:50] + '..') if len(prompt) > 50 else prompt
                  if not os.path.exists('/content/gdrive/MyDrive/Output_images/'):
                    os.mkdir('/content/gdrive/MyDrive/Output_images/')
                  if not os.path.exists('/content/gdrive/MyDrive/Output_images/' +name):
                    os.mkdir('/content/gdrive/MyDrive/Output_images/' +name)
                  r=random.randint(1,100000) 
                  filename = os.path.join(path, name, name +'_'+str(r))
                  k.save(f"{filename}.png")  
               return images
        txt2img_btn.click(t2i, [prompt, height, width, num_inference_steps, guidance_scale, batch_size], [output_txt2img_gallery])
             
 

       # generator.change(checkbox2, generator, txt2img_seed_type)	                
       # seed_btn.click(test2, alphas, generator)

demo.launch(share=True)

import gradio as gr
from frontend.css_and_js import *
from frontend.css_and_js import css
import frontend.ui_functions as uifn


def draw_gradio_ui(opt, img2img=lambda x: x, txt2img=lambda x: x, txt2img_defaults={}, RealESRGAN=True, GFPGAN=True,
                   txt2img_toggles={}, txt2img_toggle_defaults='k_euler', show_embeddings=False, img2img_defaults={},
                   img2img_toggles={}, img2img_toggle_defaults={}, sample_img2img=None, img2img_mask_modes=None,
                   img2img_resize_modes=None, user_defaults={}, run_GFPGAN=lambda x: x, run_RealESRGAN=lambda x: x):

    with gr.Blocks(css=css(opt), analytics_enabled=False, title="Stable Diffusion") as demo:
        with gr.Tabs(elem_id='tabss') as tabs:
            with gr.TabItem("Stable Diffusion Text to Image", id='txt2img_tab'):
                   
                    
                with gr.Row(elem_id='body').style(equal_height=False):
                                                          
                    with gr.Column():
                        with gr.Group(elem_id="prompt_row"):
                            output_txt2img_gallery = gr.Gallery(label="Images", elem_id="txt2img_gallery_output").style(grid=[2,3], container=True)
                            txt2img_prompt = gr.Textbox(label="Prompt", placeholder="Prompt", elem_id='prompt_input', lines=3, max_lines=4 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25, value='', show_label=False).style()
                            txt2img_btn = gr.Button("Generate", elem_id="generate", variant="primary")
                        
                    with gr.Column(elem_id="txtset"):
                        alphas=gr.Textbox(label='Seed',interactive=True, visible=False, placeholder='')
                        txt2img_width = gr.Slider(minimum=512, maximum=2048, step=64, label="Width", value=512)
                        txt2img_height = gr.Slider(minimum=512, maximum=2048, step=64, label="Height", value=512)
                        txt2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='Classifier Free Guidance Scale', value=9)
                        txt2img_steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=35)
                        txt2img_batch_count = gr.Slider(minimum=1, maximum=25, step=1, label='Number of Images', value=txt2img_defaults['n_iter'])
                        txt2img_batch_size = gr.Slider(minimum=1, maximum=7, step=1, label='Images in a Batch (memory-hungry)', value=txt2img_defaults['batch_size'], visible=False)
                        txt2img_sampling = gr.Dropdown(label='Sampling Method', choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms'], value='k_euler_a')
                        with gr.Group():
                            with gr.Row():			
                                txt2img_seed = gr.Textbox(label='Seed',interactive=True, placeholder='Random Seed')
                                txt2img_variant_seed = gr.Textbox(label="Variant Seed", placeholder="Variant Seed", lines=1, max_lines=1,value='', interactive=True)                                 
                                with gr.Column():
                                    txt2img_seed_type=gr.Checkbox(label="Random Seed", value=True)                        
                                    seed_btn = gr.Button("Keep Current Seed")
                                    txt2img_variant_amount = gr.Slider(minimum=0.0, maximum=1.0, label='Variation Amount',value=0.0, interactive=True)

 
                
                        def test2(a):
                            a=a
                            return a					
                                                    
                        def checkbox(random):
                            if random == True:
                                return gr.Textbox.update(label='Seed',interactive=True, value='', placeholder="Random Seed")
                            else:
                                return gr.Textbox.update(label='Seed', interactive=True, value='', placeholder="Random Seed")


                        def checkbox2(randoms):
                            if (randoms == ''):
                                bcc=gr.Checkbox.update(label="Random Seed", value=True)

                                return bcc

                        def checkbox3(random):
                            bc=gr.Textbox.update(interactive=True, value=alphas.value)
                            return bc

                        txt2img_seed_type.change(fn=checkbox, inputs=txt2img_seed_type, outputs=txt2img_seed)
                    
                       
                        txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=txt2img_defaults['ddim_eta'], visible=False)

     


                        output_txt2img_params = gr.Textbox(label="Generation Parameters", lines=2, interactive=False, elem_id="genparm")

                        txt2img_toggles = gr.CheckboxGroup(label='', choices=txt2img_toggles, value=txt2img_toggle_defaults, type="index", visible=False)
                        txt2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x2plus', 'RealESRGAN_x4plus_anime_6B'], value='RealESRGAN_x2plus', visible=False)
                            
                        txt2img_embeddings = gr.File(label = "Embeddings file for textual inversion", visible=False)
                    
                        output_txt2img_select_image = gr.Number(label='Image # and click Copy to copy to img2img', value=1, precision=None, visible=False)
                        with gr.Row():                           
                            output_txt2img_copy_to_input_btn = gr.Button("Send to img2img")
                            output_txt2img_stats = gr.HTML(label='Stats')                            


            

             
                       

                txt2img_btn.click(
                    txt2img,
                    [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name, txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_seed, txt2img_height, txt2img_width, txt2img_embeddings, txt2img_variant_amount, txt2img_variant_seed],
                    [output_txt2img_gallery, alphas, output_txt2img_params, output_txt2img_stats]
                )
                
                

            txt2img_seed.change(checkbox2, txt2img_seed, txt2img_seed_type)	                
            seed_btn.click(test2, alphas, txt2img_seed)


            with gr.TabItem("Stable Diffusion Image to Image", id="img2img_tab"):
                with gr.Row(elem_id="prompt_row"):
                    img2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='img2img_prompt_input',
                                                lines=1,
                                                placeholder="Prompt",
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value='',
                                                show_label=False).style()
                    img2img_btn_mask = gr.Button("Generate", variant="primary", visible=False,
                                                 elem_id="img2img_mask_btn")
                    img2img_btn_editor = gr.Button("Generate", variant="primary", elem_id="img2img_edit_btn")
                with gr.Row().style(equal_height=True):
                    with gr.Column():

                        with gr.Group():
                            img2img_image_editor = gr.Image(interactive=True, source="upload",
                                                            type="pil", tool="select", elem_id="img2img_editor",
                                                            image_mode="RGBA", visible=True)
                            img2img_image_mask = gr.Image(interactive=True, source="upload",
                                                          type="pil", tool="sketch", visible=False,
                                                          elem_id="img2img_mask")

                        



                    with gr.Column():
                        output_img2img_gallery = gr.Gallery(label="Images", elem_id="img2img_gallery_output").style(grid=[2,3], container=True)
                        with gr.Row():
                            output_img2img_copy_to_input_btn = gr.Button("To input")
                            output_img2img_copy_to_mask_btn = gr.Button("To input mask")
                            output_img2img_stats = gr.HTML(label='Stats')
                        
                        output_img2img_seed = gr.Number(label='Seed', interactive=False, visible=False)



                with gr.Row():
                    with gr.Column():
                    
                        with gr.Group():
                            with gr.Row():
                                img2img_image_editor_mode = gr.Radio(label='Editor Mode',choices=["Mask", "Crop", "Uncrop"], value="Crop", elem_id='edit_mode_select')
                                with gr.Column():
                                    with gr.Row():
                                        img2img_mask = gr.Radio(choices=["Keep Masked Area", "Regenerate Only Masked Area"],
                                                    label="Mask Mode", type="index",
                                                    value="Keep Masked Area", visible=False)

                                        img2img_painterro_btn = gr.Button("Advanced Editor", visible=True)  

               
                        
                        img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1,
                                                       label="How much blurry should the mask be? (to avoid hard edges)",
                                                       value=3, visible=False) 
 
                        img2img_width = gr.Slider(minimum=512, maximum=2048, step=64, label="Width",
                                                  value=img2img_defaults["width"])
                        img2img_height = gr.Slider(minimum=512, maximum=2048, step=64, label="Height",
                                                   value=img2img_defaults["height"])
                        
                        img2img_cfg = gr.Slider(minimum=-40.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale',
                                                value=8.0, elem_id='cfg_slider')

                        img2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1,
                                                        label='Number of Images',
                                                        value=1)
                        img2img_batch_size = gr.Slider(minimum=1, maximum=7, step=1,
                                                       label='Batch size (memory-hungry)',
                                                       value=1, visible=False)
                        img2img_dimensions_info_text_box = gr.Textbox(label="Aspect ratio (4:3 = 1.333 | 16:9 = 1.777 | 21:9 = 2.333)", visible=False)
                        
                        img2img_resize = gr.Radio(label="Resize Mode",
                                        choices=["Just resize", "Crop and resize", "Resize and fill"],
                                        type="index",
                                        value=img2img_resize_modes[img2img_defaults['resize_mode']]) 
                       
                    with gr.Column():
                        img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=50)
                                                  
                        img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength',
                                                      value=0.75)
                        
                        with gr.Group():
                            with gr.Row():			
                                img2img_seed = gr.Textbox(label='Seed',interactive=True, placeholder='Random Seed')
                                with gr.Column():
                                    img2img_seed_type=gr.Checkbox(label="Random Seed", value=True)                        
                                    imgseed_btn = gr.Button("Keep Current Seed")   
                                    
                        img2img_sampling = gr.Dropdown(label='Sampler',
                                                       choices=['k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                                                'k_heun', 'k_lms'],
                                                       value='k_euler_a')

                        output_img2img_params = gr.Textbox(label="Generation parameters")

                        img2img_toggles = gr.CheckboxGroup(label='', choices=img2img_toggles,
                                                           value=img2img_toggle_defaults, type="index", visible=False)

                        img2img_realesrgan_model_name = gr.Dropdown(choices=['RealESRGAN_x2plus',
                                                                    'RealESRGAN_x4plus_anime_6B'],
                                                                    value='RealESRGAN_x2plus',
                                                                    visible=False)  # TODO: Feels like I shouldnt slot it in here.
                        img2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=False)
                                                     
                        img2img_seed_type.change(fn=checkbox, inputs=img2img_seed_type, outputs=img2img_seed)                                                     

                img2img_image_editor_mode.change(
                    uifn.change_image_editor_mode,
                    [img2img_image_editor_mode, img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                    [img2img_image_editor, img2img_image_mask, img2img_btn_editor, img2img_btn_mask,
                     img2img_painterro_btn, img2img_mask, img2img_mask_blur_strength]
                )

                img2img_image_editor.edit(
                    uifn.update_image_mask,
                    [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                    img2img_image_mask
                )



                output_txt2img_copy_to_input_btn.click(
                    uifn.copy_img_to_input,
                    [output_txt2img_gallery],
                    [img2img_image_editor, img2img_image_mask, tabs, img2img_image_editor_mode],
                    _js=js_move_image('txt2img_gallery_output', 'img2img_editor')
                )

                output_img2img_copy_to_input_btn.click(
                    uifn.copy_img_to_edit,
                    [output_img2img_gallery],
                    [img2img_image_editor, tabs, img2img_image_editor_mode],
                    _js=js_move_image('img2img_gallery_output', 'img2img_editor')
                )
                output_img2img_copy_to_mask_btn.click(
                    uifn.copy_img_to_mask,
                    [output_img2img_gallery],
                    [img2img_image_mask, tabs, img2img_image_editor_mode],
                    _js=js_move_image('img2img_gallery_output', 'img2img_editor')
                )


                img2img_btn_mask.click(
                    img2img,
                    [img2img_prompt, img2img_image_editor_mode, img2img_image_mask, img2img_mask,
                     img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                     img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg,
                     img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                     img2img_embeddings],
                    [output_img2img_gallery, alphas, output_img2img_params, output_img2img_stats]
                )
                def img2img_submit_params():
                    return (img2img,
                    [img2img_prompt, img2img_image_editor_mode, img2img_image_editor, img2img_mask,
                     img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                     img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg,
                     img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                     img2img_embeddings],
                    [output_img2img_gallery, alphas, output_img2img_params, output_img2img_stats])
                img2img_btn_editor.click(*img2img_submit_params())
                img2img_prompt.submit(None, None, None,
                                      _js=js_img2img_submit("prompt_row"))

                img2img_painterro_btn.click(None, [img2img_image_editor], [img2img_image_editor, img2img_image_mask], _js=js_painterro_launch('img2img_editor'))

               
            img2img_seed.change(checkbox2, img2img_seed, img2img_seed_type)	                
            imgseed_btn.click(test2, alphas, img2img_seed)




            load_detector = gr.Number(value=0, label="Load Detector", visible=False)
            load_detector.change(None, None, None, _js=js(opt))
            demo.load(lambda x: 42, inputs=load_detector, outputs=load_detector)
        return demo

import re
import gradio as gr
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
from io import BytesIO
import base64
import re



def check_progress_call():

    if shared.state.job_count == 0:
        return "", gr_show(False), gr_show(False)

    progress = 0

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    progress = min(progress, 1)

    progressbar = ""
    if opts.show_progressbar:
        progressbar = f"""<div class='progressDiv'><div class='progress' style="width:{progress * 100}%">{str(int(progress*100))+"%" if progress > 0.01 else ""}</div></div>"""

    image = gr_show(False)
    preview_visibility = gr_show(False)

    if opts.show_progress_every_n_steps > 0:
        if shared.parallel_processing_allowed:

            if shared.state.sampling_step - shared.state.current_image_sampling_step >= opts.show_progress_every_n_steps and shared.state.current_latent is not None:
                shared.state.current_image = modules.sd_samplers.sample_to_image(shared.state.current_latent)
                shared.state.current_image_sampling_step = shared.state.sampling_step

        image = shared.state.current_image

        if image is None or progress >= 1:
            image = gr.update(value=None)
        else:
            preview_visibility = gr_show(True)

    return f"<span style='display: none'>{time.time()}</span><p>{progressbar}</p>", preview_visibility, image


def resize_image(resize_mode, im, width, height):
    LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width - src_w , height - src_h ))

        if ratio < src_ratio:
            fill_height = height - src_h
          #  res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width - src_w 
         #   res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res



def change_image_editor_mode(choice, cropped_image, resize_mode, width, height):
    if choice == "Mask":
        return [gr.Image.update(interactive=True, visible=False), 
                gr.Image.update(interactive=True, visible=True),
                gr.Button.update("Generate", variant="primary", visible=False),
                gr.Button.update("Generate", variant="primary", visible=True),
                gr.Button.update("Advanced Editor", visible=False),
                gr.Radio.update(choices=["Keep Masked Area", "Regenerate Only Masked Area"],
                label="Mask Mode", value="Regenerate Only Masked Area", visible=True),
                gr.Slider.update(minimum=1, maximum=10, step=1, label="How much blurry should the mask be? (to avoid hard edges)", value=3, visible=True)]
    else:
        return [gr.Image.update(visible=True), 
                gr.Image.update(visible=False),
                gr.Button.update("Generate", variant="primary", visible=True),
                gr.Button.update("Generate", variant="primary", visible=False),
                gr.Button.update("Advanced Editor", visible=True),
                gr.Radio.update(choices=["Keep Masked Area", "Regenerate Only Masked Area"],
                label="Mask Mode", value="Regenerate Only Masked Area", visible=False),
                gr.Slider.update(minimum=1, maximum=10, step=1, label="How much blurry should the mask be? (to avoid hard edges)", value=3, visible=False)]

def update_image_mask(cropped_image, resize_mode, width, height):
    resized_cropped_image = resize_image(resize_mode, cropped_image, width, height) if cropped_image else None
    return gr.Image.update(value=resized_cropped_image)

def copy_img_to_input(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.Tabs.update(selected='img2img_tab')
        img_update = gr.Image.update(value=processed_image)
        mode_update = gr.Radio.update(label='Editor Mode',choices=["Mask", "Crop", "Uncrop"], value="Crop")
        return processed_image, processed_image , tab_update, mode_update
    except IndexError:
        return [None, None]

def copy_img_to_edit(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.Tabs.update(selected='img2img_tab')
        img_update = gr.Image.update(value=processed_image)
        mode_update = gr.Radio.update(label='Editor Mode',choices=["Mask", "Crop", "Uncrop"], value="Crop")
        return processed_image, tab_update, mode_update
    except IndexError:
        return [None, None]

def copy_img_to_mask(img):
    try:
        image_data = re.sub('^data:image/.+;base64,', '', img)
        processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
        tab_update = gr.Tabs.update(selected='img2img_tab')
        img_update = gr.Image.update(value=processed_image)
        mode_update = gr.Radio.update(label='Editor Mode',choices=["Mask", "Crop", "Uncrop"], value="Mask")
        return processed_image, tab_update, mode_update
    except IndexError:
        return [None, None]



def copy_img_to_esrgan(img):
    tabs_update = gr.update(selected='esrgan')
    image_data = re.sub('^data:image/.+;base64,', '', img)
    processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return processed_image, tabs_update
    
def copy_img_to_uncrop(img):
    tabs_update = gr.update(selected='uncrop')
    image_data = re.sub('^data:image/.+;base64,', '', img)
    processed_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return processed_image, tabs_update
    
    
    
    
def copy_esrgan_to_gfpgan(output):
    try:
        update = gr.update(selected='gfpgan')
        return [output, update]
    except IndexError:
        return [None, None]

		
def copy_gfpgan_to_esrgan(output):
    try:
        update = gr.update(selected='esrgan')
        return [output, update]
    except IndexError:
        return [None, None]
		

def show_help():
    return [gr.Button.update("Show Hints", visible=False), gr.Button.update("Hide Hints", visible=True), gr.update(value=help_text)]

def hide_help():
    return [gr.Button.update("Hide Hints", visible=True), gr.Button.update("Show Hints", visible=False), gr.update(value="")]


help_text = """
    ## Mask/Crop
    * Masking is not inpainting. You will probably get better results manually masking your images in photoshop instead.
    * Built-in masking/cropping is very temperamental.
    * It may take some time for the image to show when switching from Crop to Mask.
    * If the image doesn't appear after switching to Mask, switch back to Crop and then back again to Mask
    * If the mask appears distorted (the brush is weirdly shaped instead of round), switch back to Crop and then back again to Mask.

    ## Advanced Editor
    * Click 💾 Save to send your editor changes to the img2img workflow
    * Click ❌ Clear to discard your editor changes

    If anything breaks, try switching modes again, switch tabs, clear the image, or reload.
"""

# def resize_image(resize_mode, im, width, height):
    # LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
    # if resize_mode == 0:
        # res = im.resize((width, height), resample=LANCZOS)
    # elif resize_mode == 1:
        # ratio = width / height
        # src_ratio = im.width / im.height

        # src_w = width if ratio > src_ratio else im.width * height // im.height
        # src_h = height if ratio <= src_ratio else im.height * width // im.width

        # resized = im.resize((src_w, src_h), resample=LANCZOS)
        # res = Image.new("RGBA", (width, height))
        # res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    # else:
        # ratio = width / height
        # src_ratio = im.width / im.height

        # src_w = width if ratio < src_ratio else im.width * height // im.height
        # src_h = height if ratio >= src_ratio else im.height * width // im.width

        # resized = im.resize((src_w, src_h), resample=LANCZOS)
        # res = Image.new("RGBA", (width, height))
        # res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        # if ratio < src_ratio:
            # fill_height = height // 2 - src_h // 2
            # res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            # res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        # elif ratio > src_ratio:
            # fill_width = width // 2 - src_w // 2
            # res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            # res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    # return res
    


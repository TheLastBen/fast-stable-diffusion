import html
from modules import script_callbacks, shared, scripts
import gradio as gr
from pathlib import Path
import os
import sys
import platform
from launch import run

import string
import random as rd

key_characters = (string.ascii_letters + string.digits)


def random_string(length=20):
    return ''.join([rd.choice(key_characters) for _ in range(length)])

key = random_string()


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        #refresh = gr.Button(value="refresh", variant="primary")
        canvas = gr.HTML(
            f"<iframe id=\"openoutpaint-iframe\" class=\"border-2 border-gray-200\" src=\"{scripts.basedir()}/{usefulDirs[0]}/{usefulDirs[1]}/app/index.html?noprompt\"></iframe>")
        keyinput = gr.HTML(
            f"<input id=\"openoutpaint-key\" type=\"hidden\" value=\"{key}\">")
        # refresh.click(

        # )

    return [(ui, "openOutpaint", "openOutpaint")]


usefulDirs = scripts.basedir().split(os.sep)[-2:]

print("openOutpaint init")
git = os.environ.get('GIT', "git")
print(run(f'"{git}" -C "' + scripts.basedir() +
      '" submodule update --init --recursive --remote'))

with open(f"{scripts.basedir()}/app/key.json", "w") as keyfile:
    keyfile.write('{\n')
    keyfile.write(f"  \"key\": \"{key}\"\n")
    keyfile.write('}\n')
    keyfile.close()

script_callbacks.on_ui_tabs(add_tab)

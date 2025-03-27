import gradio as gr
import torch
import os
import glob
import spaces
import numpy as np

from PIL import Image
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from ConsistentID.lib.pipeline_ConsistentID import ConsistentIDPipeline
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', type=str, 
                    default="models/Realistic_Vision_V4.0_noVAE")
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = f"cuda:{args.gpu}"

### Load base model
pipe = ConsistentIDPipeline.from_pretrained(
    args.base_model_path, 
    torch_dtype=torch.float16, 
)

### Load consistentID_model checkpoint
pipe.load_ConsistentID_model(
    consistentID_weight_path="./models/ConsistentID-v1.bin",
    bise_net_weight_path="./models/BiSeNet_pretrained_for_ConsistentID.pth",
)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device, torch.float16)

@spaces.GPU
def process(selected_template_images, custom_image, prompt, 
            negative_prompt, prompt_selected, model_selected_tab, 
            prompt_selected_tab, guidance_scale, width, height, merge_steps, seed_set):
    
    # The gradio UI only supports one image at a time.
    if model_selected_tab==0:
        subj_images = load_image(Image.open(selected_template_images))
    else:
        subj_images = load_image(Image.fromarray(custom_image))

    if prompt_selected_tab==0:
        prompt = prompt_selected
        negative_prompt = ""

    # hyper-parameter
    num_steps = 50
    seed_set = torch.randint(0, 1000, (1,)).item()
    # merge_steps = 30
            
    if prompt == "":
        prompt = "A man, in a forest"
        prompt = "A man, with backpack, in a raining tropical forest, adventuring, holding a flashlight, in mist, seeking animals"
        prompt = "A person, in a sowm, wearing santa hat and a scarf, with a cottage behind"
    else:
        #prompt=Enhance_prompt(prompt, Image.new('RGB', (200, 200), color = 'white'))
        print(prompt)

    if negative_prompt == "":
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    #Extend Prompt
    #prompt = "cinematic photo," + prompt + ", 50mm photograph, half-length portrait, film, bokeh, professional, 4k, highly detailed"
    #print(prompt)

    negtive_prompt_group="((cross-eye)),((cross-eyed)),(((NFSW))),(nipple),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
    negative_prompt = negative_prompt + negtive_prompt_group
    
    # seed = torch.randint(0, 1000, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed_set)

    images = pipe(
        prompt=prompt,
        width=width,    
        height=height,
        input_subj_image_objs=subj_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        start_merge_step=merge_steps,
        generator=generator,
    ).images[0]

    return np.array(images)

# Gets the templates
preset_template = glob.glob("./images/templates/*.png")
preset_template = preset_template + glob.glob("./images/templates/*.jpg")

with gr.Blocks(title="ConsistentID Demo") as demo:
    gr.Markdown("# ConsistentID Demo")
    gr.Markdown("\
        Put the reference figure to be redrawn into the box below (There is a small probability of referensing failure. You can submit it repeatedly)")
    gr.Markdown("\
        If you find our work interesting, please leave a star in GitHub for us!<br>\
        https://github.com/JackAILab/ConsistentID")
    with gr.Row():
        with gr.Column():
            model_selected_tab = gr.State(0)
            with gr.TabItem("template images") as template_images_tab:
                template_gallery_list = [(i, i) for i in preset_template]
                gallery = gr.Gallery(template_gallery_list,columns=[4], rows=[2], object_fit="contain", height="auto",show_label=False)
                
                def select_function(evt: gr.SelectData):
                    return preset_template[evt.index]

                selected_template_images = gr.Text(show_label=False, visible=False, placeholder="Selected")
                gallery.select(select_function, None, selected_template_images)
            with gr.TabItem("Upload Image") as upload_image_tab:
                custom_image = gr.Image(label="Upload Image")

            model_selected_tabs = [template_images_tab, upload_image_tab]
            for i, tab in enumerate(model_selected_tabs):
                tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[model_selected_tab])

            with gr.Column():
                prompt_selected_tab = gr.State(0)
                with gr.TabItem("template prompts") as template_prompts_tab:
                    prompt_selected = gr.Dropdown(value="A person, police officer, half body shot", elem_id='dropdown', choices=[
                        "A woman in a wedding dress",
                        "A woman, queen, in a gorgeous palace",
                        "A man sitting at the beach with sunset", 
                        "A person, police officer, half body shot", 
                        "A man, sailor, in a boat above ocean",
                        "A women wearing headphone, listening music", 
                        "A man, firefighter, half body shot"], label=f"prepared prompts")

                with gr.TabItem("custom prompt") as custom_prompt_tab:
                    prompt = gr.Textbox(label="prompt",placeholder="A man/woman wearing a santa hat")
                    nagetive_prompt = gr.Textbox(label="negative prompt",placeholder="monochrome, lowres, bad anatomy, worst quality, low quality, blurry")
            
                prompt_selected_tabs = [template_prompts_tab, custom_prompt_tab]
                for i, tab in enumerate(prompt_selected_tabs):
                    tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[prompt_selected_tab])

            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=1.0,
                maximum=10.0,
                step=1.0,
                value=5.0,
            )

            width = gr.Slider(label="image width",minimum=256,maximum=768,value=512,step=8)
            height = gr.Slider(label="image height",minimum=256,maximum=768,value=512,step=8)
            width.release(lambda x,y: min(1280-x,y), inputs=[width,height], outputs=[height])
            height.release(lambda x,y: min(1280-y,x), inputs=[width,height], outputs=[width])
            merge_steps = gr.Slider(label="step starting to merge facial details(30 is recommended)",minimum=10,maximum=50,value=30,step=1)
            seed_set = gr.Slider(label="set the random seed for different results",minimum=1,maximum=2147483647,value=2024,step=1)
            
            btn = gr.Button("Run")
        with gr.Column():
            out = gr.Image(label="Output")
            gr.Markdown('''
                N.B.:<br/>
                - If the proportion of face in the image is too small, the probability of an error will be slightly higher, and the similarity will also significantly decrease.)
                - At the same time, use prompt with \"man\" or \"woman\" instead of \"person\" as much as possible, as that may cause the model to be confused whether the protagonist is male or female.
                - Due to insufficient graphics memory on the demo server, there is an upper limit on the resolution for generating samples. We will support the generation of SDXL as soon as possible<br/><br/>
                ''')
        btn.click(fn=process, inputs=[selected_template_images, custom_image,prompt, nagetive_prompt, prompt_selected,
                                      model_selected_tab, prompt_selected_tab, guidance_scale, width, height, merge_steps, seed_set], outputs=out)

demo.launch(server_name='0.0.0.0', ssl_verify=False)

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import cv2
import PIL
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms
from insightface.app import FaceAnalysis 
### insight-face installation can be found at https://github.com/deepinsight/insightface
from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from .functions import insert_markers_to_prompt, masks_for_unique_values, apply_mask_to_raw_image, tokenize_and_mask_noun_phrases_ends, prepare_image_token_idx
from .functions import ProjPlusModel, masks_for_unique_values
from .attention import Consistent_IPAttProcessor, Consistent_AttProcessor, FacialEncoder
from easydict import EasyDict as edict
from huggingface_hub import hf_hub_download
### Model can be imported from https://github.com/zllrunning/face-parsing.PyTorch?tab=readme-ov-file
### We use the ckpt of 79999_iter.pth: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
### Thanks for the open source of face-parsing model.
from .BiSeNet.model import BiSeNet
import os

PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]

### Download the pretrained model from huggingface and put it locally, then place the model in a local directory and specify the directory location.
class ConsistentIDPipeline(StableDiffusionPipeline):
    # to() should be only called after all modules are loaded.
    def to(
        self,
        torch_device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().to(torch_device,                dtype=dtype)
        self.bise_net.to(torch_device,          dtype=dtype)
        self.clip_encoder.to(torch_device,      dtype=dtype)
        self.image_proj_model.to(torch_device,  dtype=dtype)
        self.FacialEncoder.to(torch_device,     dtype=dtype)
        # If the unet is not released, the ip_layers should be moved to the specified device and dtype.
        if not isinstance(self.unet, edict):
            self.ip_layers.to(torch_device,     dtype=dtype)
        return self

    @validate_hf_hub_args
    def load_ConsistentID_model(
        self,
        consistentID_weight_path:   str,
        bise_net_weight_path:       str,
        trigger_word_facial:        str = '<|facial|>',
        # A CLIP ViT-H/14 model trained with the LAION-2B English subset of LAION-5B using OpenCLIP.
        # output dim: 1280.
        image_encoder_path:         str = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',  
        torch_dtype = torch.float16,
        num_tokens = 4,
        lora_rank= 128,
        **kwargs,
    ):
        self.lora_rank = lora_rank 
        self.torch_dtype = torch_dtype
        self.num_tokens = num_tokens
        self.set_ip_adapter()
        self.image_encoder_path = image_encoder_path
        self.clip_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path)
        self.clip_preprocessor  = CLIPImageProcessor()
        self.id_image_processor = CLIPImageProcessor()
        self.crop_size = 512

        # face_app: FaceAnalysis object
        self.face_app = FaceAnalysis(name="buffalo_l", root='models/insightface', providers=['CPUExecutionProvider'])
        # The original det_size=(640, 640) is too large and face_app often fails to detect faces.
        self.face_app.prepare(ctx_id=0, det_size=(512, 512))

        if not os.path.exists(consistentID_weight_path):
            ### Download pretrained models
            hf_hub_download(repo_id="JackAILab/ConsistentID", repo_type="model",
                            filename=os.path.basename(consistentID_weight_path), 
                            local_dir=os.path.dirname(consistentID_weight_path))
        if not os.path.exists(bise_net_weight_path):
            hf_hub_download(repo_id="JackAILab/ConsistentID", 
                            filename=os.path.basename(bise_net_weight_path), 
                            local_dir=os.path.dirname(bise_net_weight_path))

        bise_net = BiSeNet(n_classes = 19)
        bise_net.load_state_dict(torch.load(bise_net_weight_path, map_location="cpu"))
        bise_net.eval()
        self.bise_net = bise_net

        # Colors for all 20 parts
        self.part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                            [255, 0, 85], [255, 0, 170],
                            [0, 255, 0], [85, 255, 0], [170, 255, 0],
                            [0, 255, 85], [0, 255, 170],
                            [0, 0, 255], [85, 0, 255], [170, 0, 255],
                            [0, 85, 255], [0, 170, 255],
                            [255, 255, 0], [255, 255, 85], [255, 255, 170],
                            [255, 0, 255], [255, 85, 255], [255, 170, 255],
                            [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        # image_proj_model maps 1280-dim OpenCLIP embeddings to 768-dim face prompt embeddings.
        self.image_proj_model = ProjPlusModel(
            cross_attention_dim=self.unet.config.cross_attention_dim, 
            id_embeddings_dim=512,
            clip_embeddings_dim=self.clip_encoder.config.hidden_size, 
            num_tokens=self.num_tokens,  # 4 - inspirsed by IPAdapter and Midjourney
        )
        self.FacialEncoder = FacialEncoder()

        if consistentID_weight_path.endswith(".safetensors"):
            state_dict = {"id_encoder": {}, "lora_weights": {}}
            with safe_open(consistentID_weight_path, framework="pt", device="cpu") as f:
                ### TODO safetensors add
                for key in f.keys():
                    if key.startswith("FacialEncoder."):
                        state_dict["FacialEncoder"][key.replace("FacialEncoder.", "")] = f.get_tensor(key)
                    elif key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(consistentID_weight_path, map_location="cpu")
            
        self.trigger_word_facial = trigger_word_facial

        self.FacialEncoder.load_state_dict(state_dict["FacialEncoder"], strict=True)
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        self.ip_layers.load_state_dict(state_dict["adapter_modules"], strict=True)
        print(f"Successfully loaded weights from checkpoint")

        # Add trigger word token
        if self.tokenizer is not None: 
            self.tokenizer.add_tokens([self.trigger_word_facial], special_tokens=True)

    def set_ip_adapter(self):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = Consistent_AttProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                )
            else:
                attn_procs[name] = Consistent_IPAttProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank, num_tokens=self.num_tokens,
                )
        
        unet.set_attn_processor(attn_procs)

    @torch.inference_mode()
    # parsed_image_parts2 is a batched tensor of parsed_image_parts with bs=1. It only contains the facial areas of one input image.
    # clip_encoder maps image parts to image-space diffusion prompts.
    # Then the facial class token embeddings are replaced with the fused (multi_facial_embeds, prompt_embeds[class_tokens_mask]).
    def extract_local_facial_embeds(self, prompt_embeds, uncond_prompt_embeds, parsed_image_parts2, 
                                    facial_token_masks, valid_facial_token_idx_mask, calc_uncond=True):
        
        hidden_states = []
        uncond_hidden_states = []
        for parsed_image_parts in parsed_image_parts2:
            hidden_state = self.clip_encoder(parsed_image_parts.to(self.device, dtype=self.torch_dtype), output_hidden_states=True).hidden_states[-2]
            uncond_hidden_state = self.clip_encoder(torch.zeros_like(parsed_image_parts, dtype=self.torch_dtype).to(self.device), output_hidden_states=True).hidden_states[-2]
            hidden_states.append(hidden_state)
            uncond_hidden_states.append(uncond_hidden_state)
        multi_facial_embeds = torch.stack(hidden_states)       
        uncond_multi_facial_embeds = torch.stack(uncond_hidden_states)   

        # conditional prompt.
        # FacialEncoder maps multi_facial_embeds to facial ID embeddings, and replaces the class tokens in prompt_embeds 
        # with the fused (facial ID embeddings, prompt_embeds[class_tokens_mask]).
        # multi_facial_embeds: [1, 5, 257, 1280].
        facial_prompt_embeds = self.FacialEncoder(prompt_embeds, multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask)  

        if not calc_uncond:
            return facial_prompt_embeds, None
        # unconditional prompt.
        uncond_facial_prompt_embeds = self.FacialEncoder(uncond_prompt_embeds, uncond_multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask)  

        return facial_prompt_embeds, uncond_facial_prompt_embeds        

    @torch.inference_mode()
    # Extrat OpenCLIP embeddings from the input image and map them to face prompt embeddings.
    def extract_global_id_embeds(self, face_image_obj, s_scale=1.0, shortcut=False):
        clip_image_ts = self.clip_preprocessor(images=face_image_obj, return_tensors="pt").pixel_values
        clip_image_ts = clip_image_ts.to(self.device, dtype=self.torch_dtype)
        clip_image_embeds = self.clip_encoder(clip_image_ts, output_hidden_states=True).hidden_states[-2]
        uncond_clip_image_embeds = self.clip_encoder(torch.zeros_like(clip_image_ts), output_hidden_states=True).hidden_states[-2]

        faceid_embeds = self.extract_faceid(face_image_obj)
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        # image_proj_model maps 1280-dim OpenCLIP embeddings to 768-dim face prompt embeddings.
        # clip_image_embeds are used as queries to transform faceid_embeds.
        # faceid_embeds -> kv, clip_image_embeds -> q
        global_id_embeds        = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)
        uncond_global_id_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=shortcut, scale=s_scale)
        
        return global_id_embeds, uncond_global_id_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, Consistent_IPAttProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def extract_faceid(self, face_image_obj):
        faceid_image = np.array(face_image_obj)
        faces = self.face_app.get(faceid_image)
        if faces==[]:
            faceid_embeds = torch.zeros_like(torch.empty((1, 512)))
        else:
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        return faceid_embeds

    @torch.inference_mode()
    def parse_face_mask(self, raw_image_refer):

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        to_pil = transforms.ToPILImage()

        with torch.no_grad():
            image = raw_image_refer.resize((512, 512), Image.BILINEAR)
            image_resize_PIL = image
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device, dtype=self.torch_dtype)
            out = self.bise_net(img)[0]
            parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)
        
        im = np.array(image_resize_PIL)
        vis_im = im.copy().astype(np.uint8)
        stride=1
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1): # num_of_class=17 pi=1~16
            index = np.where(vis_parsing_anno == pi) 
            vis_parsing_anno_color[index[0], index[1], :] = self.part_colors[pi] 

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_parsing_anno_color = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        return vis_parsing_anno_color, vis_parsing_anno

    @torch.inference_mode()
    def extract_facemask(self, input_image_obj):
        vis_parsing_anno_color, vis_parsing_anno = self.parse_face_mask(input_image_obj)
        parsing_mask_list = masks_for_unique_values(vis_parsing_anno) 

        key_parsing_mask_dict = {}
        key_list = ["Face", "Left_Ear", "Right_Ear", "Left_Eye", "Right_Eye", "Nose", "Upper_Lip", "Lower_Lip"]
        processed_keys = set()
        for key, mask_image in parsing_mask_list.items():
            if key in key_list:
                if "_" in key:
                    prefix = key.split("_")[1]
                    if prefix in processed_keys:                   
                        continue
                    else:            
                        key_parsing_mask_dict[key] = mask_image 
                        processed_keys.add(prefix)  
            
                key_parsing_mask_dict[key] = mask_image            

        return key_parsing_mask_dict, vis_parsing_anno_color

    def augment_prompt_with_trigger_word(
        self,
        prompt: str,
        face_caption: str,
        key_parsing_mask_dict = None,
        facial_token = "<|facial|>",
        max_num_facials = 5,
        num_id_images: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        # face_caption_align: 'The person has one nose <|facial|>, two ears <|facial|>, two eyes <|facial|>, and a mouth <|facial|>, '
        face_caption_align, key_parsing_mask_dict_align = insert_markers_to_prompt(face_caption, key_parsing_mask_dict) 
        
        prompt_face = prompt + " Detail: " + face_caption_align

        max_text_length=330      
        if len(self.tokenizer(prompt_face, max_length=self.tokenizer.model_max_length, 
                              padding="max_length", truncation=False, return_tensors="pt").input_ids[0]) != 77:
            # Put face_caption_align at the beginning of the prompt, so that the original prompt is truncated,
            # but the face_caption_align is well kept.
            prompt_face = "Detail: " + face_caption_align + " Caption:" + prompt

        # Remove "<|facial|>" from prompt_face.
        # augmented_prompt: 'A person, police officer, half body shot Detail: 
        # The person has one nose , two ears , two eyes , and a mouth , '
        augmented_prompt = prompt_face.replace("<|facial|>", "")
        tokenizer = self.tokenizer
        facial_token_id = tokenizer.convert_tokens_to_ids(facial_token)
        image_token_id = None

        # image_token_id: the token id of "<|image|>". Disabled, as it's set to None.
        # facial_token_id: the token id of "<|facial|>".
        clean_input_id, image_token_mask, facial_token_mask = \
            tokenize_and_mask_noun_phrases_ends(prompt_face, image_token_id, facial_token_id, tokenizer) 

        image_token_idx, image_token_idx_mask, facial_token_idx, facial_token_idx_mask = \
            prepare_image_token_idx(image_token_mask, facial_token_mask, num_id_images, max_num_facials)

        return augmented_prompt, clean_input_id, key_parsing_mask_dict_align, facial_token_mask, facial_token_idx, facial_token_idx_mask

    @torch.inference_mode()
    def extract_parsed_image_parts(self, input_image_obj, key_parsing_mask_dict, image_size=512, max_num_facials=5):
        facial_masks = []
        parsed_image_parts = []
        key_masked_raw_images_dict = {}
        transform_mask = transforms.Compose([transforms.CenterCrop(size=image_size), transforms.ToTensor(),])
        clip_preprocessor = CLIPImageProcessor()

        num_facial_part = len(key_parsing_mask_dict)

        for key in key_parsing_mask_dict:
            key_mask=key_parsing_mask_dict[key]
            facial_masks.append(transform_mask(key_mask))
            key_masked_raw_image = apply_mask_to_raw_image(input_image_obj, key_mask)
            key_masked_raw_images_dict[key] = key_masked_raw_image
            # clip_preprocessor normalizes key_masked_raw_image, so that (masked) zero pixels become non-zero.
            # It also resizes the image to 224x224.
            parsed_image_part = clip_preprocessor(images=key_masked_raw_image, return_tensors="pt").pixel_values
            parsed_image_parts.append(parsed_image_part)
            
        padding_ficial_clip_image = torch.zeros_like(torch.zeros([1, 3, 224, 224]))
        padding_ficial_mask = torch.zeros_like(torch.zeros([1, image_size, image_size]))

        if num_facial_part < max_num_facials:
            parsed_image_parts  += [ torch.zeros_like(padding_ficial_clip_image) for _ in range(max_num_facials - num_facial_part) ]
            facial_masks        += [ torch.zeros_like(padding_ficial_mask)       for _ in range(max_num_facials - num_facial_part) ]

        parsed_image_parts  = torch.stack(parsed_image_parts, dim=1).squeeze(0)
        facial_masks        = torch.stack(facial_masks,       dim=0).squeeze(dim=1)

        return parsed_image_parts, facial_masks, key_masked_raw_images_dict

    # Release the unet/vae/text_encoder to save memory.
    def release_components(self, released_components=["unet", "vae", "text_encoder"]):
        if "unet" in released_components:
            unet = edict()
            # Only keep the config and in_channels attributes that are used in the pipeline.
            unet.config = self.unet.config
            self.unet = unet

        if "vae" in released_components:
            self.vae = None
        if "text_encoder" in released_components:
            self.text_encoder = None

    # input_subj_image_obj: an Image object.
    def extract_double_id_prompt_embeds(self, prompt, negative_prompt, input_subj_image_obj, device, calc_uncond=True):
        face_caption = "The person has one nose, two eyes, two ears, and a mouth."
        key_parsing_mask_dict, vis_parsing_anno_color = self.extract_facemask(input_subj_image_obj)

        augmented_prompt, clean_input_id, key_parsing_mask_dict_align, \
          facial_token_mask, facial_token_idx, facial_token_idx_mask \
            = self.augment_prompt_with_trigger_word(
                prompt = prompt,
                face_caption = face_caption,
                key_parsing_mask_dict=key_parsing_mask_dict,
                device=device,
                max_num_facials = 5,
                num_id_images = 1
                )

        text_embeds, uncond_text_embeds = self.encode_prompt(
            augmented_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=calc_uncond,
            negative_prompt=negative_prompt,
        )

        # 5. Prepare the input ID images
        # global_id_embeds: [1, 4, 768]
        # extract_global_id_embeds() extrats OpenCLIP embeddings from the input image and map them to global face prompt embeddings.
        global_id_embeds, uncond_global_id_embeds = \
            self.extract_global_id_embeds(face_image_obj=input_subj_image_obj, s_scale=1.0, shortcut=False)

        # parsed_image_parts: [5, 3, 224, 224]. 5 parts, each part is a 3-channel 224x224 image (resized by CLIP Preprocessor).
        parsed_image_parts, facial_masks, key_masked_raw_images_dict = \
            self.extract_parsed_image_parts(input_subj_image_obj, key_parsing_mask_dict_align, image_size=512, max_num_facials=5)
        parsed_image_parts2 = parsed_image_parts.unsqueeze(0).to(device, dtype=self.torch_dtype)
        facial_token_mask = facial_token_mask.to(device)
        facial_token_idx_mask = facial_token_idx_mask.to(device)

        # key_masked_raw_images_dict: ['Right_Eye', 'Right_Ear', 'Nose', 'Upper_Lip']
        # for key in key_masked_raw_images_dict:
        #     key_masked_raw_images_dict[key].save(f"{key}.png")

        # 6. Get the update text embedding
        # parsed_image_parts2: the facial areas of the input image
        # extract_local_facial_embeds() maps parsed_image_parts2 to multi_facial_embeds, and then replaces the class tokens in prompt_embeds 
        # with the fused (id_embeds, prompt_embeds[class_tokens_mask]) whose indices are specified by class_tokens_mask.        
        # parsed_image_parts2: [1, 5, 3, 224, 224]
        text_local_id_embeds, uncond_text_local_id_embeds = \
            self.extract_local_facial_embeds(text_embeds, uncond_text_embeds, \
                                             parsed_image_parts2, facial_token_mask, facial_token_idx_mask,
                                             calc_uncond=calc_uncond)

        # text_global_id_embeds, text_local_global_id_embeds: [1, 81, 768]
        # text_local_id_embeds: [1, 77, 768], only differs with text_embeds on 4 ID embeddings, and is identical
        # to text_embeds on the rest 73 tokens.
        text_global_id_embeds         = torch.cat([text_embeds,          global_id_embeds], dim=1)
        text_local_global_id_embeds   = torch.cat([text_local_id_embeds, global_id_embeds], dim=1)

        if calc_uncond:
            uncond_text_global_id_embeds  = torch.cat([uncond_text_local_id_embeds, uncond_global_id_embeds], dim=1)
            coarse_prompt_embeds = torch.cat([uncond_text_global_id_embeds, text_global_id_embeds], dim=0)
            fine_prompt_embeds   = torch.cat([uncond_text_global_id_embeds, text_local_global_id_embeds], dim=0)
        else:
            coarse_prompt_embeds = text_global_id_embeds
            fine_prompt_embeds   = text_local_global_id_embeds

        # fine_prompt_embeds: the conditional part is 
        # (text_global_id_embeds + text_local_global_id_embeds) / 2.
        fine_prompt_embeds   = (coarse_prompt_embeds + fine_prompt_embeds) / 2

        return coarse_prompt_embeds, fine_prompt_embeds
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        input_subj_image_objs: PipelineImageInput = None,
        start_merge_step: int = 0,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale >= 1.0
        assert do_classifier_free_guidance

        if input_subj_image_objs is not None:
            if not isinstance(input_subj_image_objs, list):
                input_subj_image_objs = [input_subj_image_objs]

            # 3. Encode input prompt
            coarse_prompt_embeds, fine_prompt_embeds = \
                self.extract_double_id_prompt_embeds(prompt, negative_prompt, input_subj_image_objs[0], device)
        else:
            # Replace the coarse_prompt_embeds and fine_prompt_embeds with the input prompt_embeds.
            # This is used when prompt_embeds are computed in advance.
            cfg_prompt_embeds    = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            coarse_prompt_embeds = cfg_prompt_embeds
            fine_prompt_embeds   = cfg_prompt_embeds

        # 7. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 8. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.dtype,
            device,
            generator,
            latents,
        )

        # {'eta': 0.0, 'generator': None}. eta is 0 for DDIM.
        extra_step_kwargs    = self.prepare_extra_step_kwargs(generator, eta)
        cross_attention_kwargs = {}

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                # DDIM doesn't scale latent_model_input.                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                if i <= start_merge_step:
                    current_prompt_embeds = coarse_prompt_embeds
                else:
                    current_prompt_embeds = fine_prompt_embeds

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    assert 0, "Not Implemented"

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or \
                  ( (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0 ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 9.1 Post-processing
            image = self.decode_latents(latents)
            # 9.3 Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 9.1 Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=None
        )









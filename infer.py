import sys,os
import numpy as np
from PIL import Image,ImageDraw,ImageFont
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import argparse
import logging
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import diffusers
import json
from src.dataloader import Subjects200K,collate_fn
from omegaconf import OmegaConf
from glob import glob
import functools
import cv2
import importlib
logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

try:
    font = ImageFont.truetype("arial.ttf", 32)
except IOError:
    font = ImageFont.load_default()
    
def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="testing script.")
    parser.add_argument("--basemodel", type=str,default="UniGenFlux",)
    parser.add_argument("--pipeline", type=str,default="UniGenFLUXPipeline",)
    parser.add_argument("--pretrained_model_name_or_path", type=str,default="ckpt/FLUX.1-schnell",)
    parser.add_argument("--transformer",type=str,default="ckpt/FLUX.1-schnell/transformer",)
    parser.add_argument("--cn_config", type=str, default='cfgs/sd35_control.yaml')

    parser.add_argument("--disable_single_trans_blocks",action="store_true",default=False,)
    parser.add_argument("--single_block_control_method",type=str,default="overall_add",)
    parser.add_argument("--single_control_dev",type=int,default=2,)

    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--condition_types", type=str, nargs='+', default=["canny", "depth"], )
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers",type=int,default=0,)
    
    parser.add_argument("--max_sequence_length",type=int,default=512,help="Maximum sequence length to use with with the T5 text encoder")
    parser.add_argument("--guidance_scale",type=float,default=3.5,help="guidance scale") # Flux: 3.5
    parser.add_argument("--num_inference_steps",type=int,default=28,help="inference denoise steps") # Flux: 50
    
    parser.add_argument("--work_dir",type=str,default="output/test_result")

    parser.add_argument("--cache_dir",type=str,default="cache")
    parser.add_argument("--seed", type=int, default=12443)
    parser.add_argument("--mixed_precision",type=str,default="bf16",choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed running: local_rank")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    args.revision = None
    args.variant = None
    return args

def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 2. set seed
    if args.seed is not None:
        set_seed(args.seed)

    # load total batch size
    args.total_batch_size=args.batch_size * accelerator.num_processes

    # 3. create the working directory
    if accelerator.is_main_process:
        if args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)

    # 4. precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 5. Load pretrained single conditional LoRA modules onto the FLUX transformer
    cn_config=OmegaConf.load(args.cn_config)
    cn_config.params.control_params.use_single_trans_blocks=not args.disable_single_trans_blocks
    cn_config.params.control_params.single_block_control_method=args.single_block_control_method
    cn_config.params.control_params.single_control_dev=args.single_control_dev
    transformer = getattr(importlib.import_module("src.UniGenTransformer"),args.basemodel).from_pretrained(
        pretrained_model_name_or_path=f'{args.pretrained_model_name_or_path}/transformer',
        revision=args.revision,
        variant=args.variant
    ).to(accelerator.device, dtype=weight_dtype)
    cn_config.params.control_params.use_transformer_params=False
    transformer.init_condition_block(condition_nums=len(args.condition_types),condition_types=args.condition_types,**cn_config.params)

    # load pretrain ckpts
    if os.path.exists(f"{args.transformer}/latest"):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        steps=open(f"{args.transformer}/latest","r").read().strip()
        load_state_dict = get_fp32_state_dict_from_zero_checkpoint(args.transformer)
        load_res=transformer.load_state_dict(load_state_dict,strict=False)
        args.work_dir=f'{args.work_dir}/{steps}'
        if accelerator.is_main_process:
            os.makedirs(args.work_dir, exist_ok=True)
    elif os.path.exists(args.transformer):
        load_state_dict=torch.load(args.transformer)
        load_res=transformer.load_state_dict(load_state_dict,strict=False)
    else:
        from safetensors.torch import load_file
        load_state_dict=dict()
        for file in glob(os.path.join(args.transformer,'*.safetensors')):
            load_state_dict.update(load_file(file))
        load_res=transformer.load_state_dict(load_state_dict,strict=False)
    logger.info(f"load pretrain ckpts success, load result: {load_res}")
    
    transformer.requires_grad_(False)

    # 6. get the inference pipeline.
    pipe = getattr(importlib.import_module("src.UniGenPipeline"),args.pipeline).from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
    ).to(accelerator.device, dtype=weight_dtype)
    pipe.transformer = transformer

    # pipe.eval()
    pipe.to(accelerator.device, dtype=weight_dtype)

    # 7. get the VAE image processor to do the pre-process and post-process for images.
    # (vae_scale_factor is the scale of downsample.)
    vae_scale_factor = pipe.vae_scale_factor
    
    # 8. get the dataset
    dataset=Subjects200K(args.data_path,args.condition_types,pipe.vae_scale_factor,resolution=args.resolution,split="test",test_split='depth_subject_pose.txt')
    logger.info(f'Load {args.condition_types} train Subjects200K dataset success, data length: {len(dataset)}')

    # 10. get the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=functools.partial(collate_fn,split="test"),
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # 11. accelerator start
    pipe, dataloader = accelerator.prepare(
        pipe, dataloader
    )

    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Transformer Class = {transformer.__class__.__name__}")
    logger.info(f"  Num of GPU processes = {accelerator.num_processes}")


    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    logger.info(f"generate sample save dir: {args.work_dir}")
    # os.makedirs(os.path.join(output_dir, "info"), exist_ok=True)

    # 11. start testing!
    for S, batch in enumerate(dataloader):
        prompts = batch["descriptions"]
        cond_imgs=batch['condition_img']
        cond_prompts=batch['task_names']
        
        prompts=[prompt.replace('<|endoftext|>','').replace("!","") for prompt in prompts]
        
        with torch.no_grad():
            result_img_list = pipe(
                prompt=prompts,
                condition_prompt=cond_prompts,
                control_image=cond_imgs,
                # conditioning_scale=args.condition_scale,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
                dtype=transformer.dtype,
                generator=torch.Generator("cpu").manual_seed(args.seed)
            ).images

        # 12. save the output to the output_dir = args.work_dir
        for i, (target_img,res_img,cond_img,prompt,cond_prompt,batch_id) in enumerate(zip(batch['target_img'],result_img_list,cond_imgs,prompts,cond_prompts,batch['batch_ids'])):
            base_path=f'{args.work_dir}/{batch_id}_{cond_prompt}'
            os.makedirs(base_path,exist_ok=True)
            
            target_img.save(f'{base_path}/target.png')
            res_img.save(f'{base_path}/res.png')
            cond_img.save(f'{base_path}/condition.png')
            
            with open(os.path.join(base_path,"info.json"), "w", encoding="utf-8") as file:
                meta_data = {
                    "description": prompt,
                    "condition_prompt":cond_prompt
                }
                json.dump(meta_data,file, ensure_ascii=False, indent=4)
        
        progress_bar.update(1)

    total_params=sum(p.numel() for p in pipe.transformer.parameters())
    print(f'In condition types: {args.condition_types}, load trained denoise adapter. Total params: {format_params(total_params)}')

def format_params(num_params):
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)
    
if __name__ == "__main__":
    
    args = parse_args()
    with torch.no_grad():
        main(args)
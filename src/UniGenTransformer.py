import numpy as np
import tqdm
from src.UniGenUtils import MoE, modulated_flatten,zero_module,debug_print
import torch
from diffusers.configuration_utils import register_to_config
from typing import Any, Dict, Optional, Union, List, Tuple
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock,FluxSingleTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND,scale_lora_layers,unscale_lora_layers
from torch import Value, nn
from .lora_switching_module import enable_lora,module_active_adapters
import copy
import logging
import os
from diffusers import FluxTransformer2DModel,SanaTransformer2DModel,SD3Transformer2DModel
from torch.utils.checkpoint import checkpoint
from diffusers.models.embeddings import PatchEmbed,CombinedTimestepTextProjEmbeddings
from src.UniGenUtils import *

logger = logging.getLogger(__name__)

class UniGenBase(SD3Transformer2DModel):
    def init_condition_block(self,condition_nums=1,**kwargs): 
        self.condition_nums=condition_nums   
        self.init_control_block(kwargs.get('control_params',None))
    
    def init_control_block(self, control_params=None):
        assert control_params is not None, ValueError('Please provice control net model parameter')
        self.trainable_control_modules=dict()
        self.use_pooled_prompt_embeds=control_params.get('use_pooled_prompt_embeds',True)
        self.use_encoder_hidden_states=control_params.get('use_encoder_hidden_states',True)
        self.control_blocks_num=control_params.get('num_layers',self.config.num_layers)
        self.use_rope=control_params.get('use_rope',False)
        
        num_attention_heads,attention_head_dim=control_params.get('num_attention_heads',self.config.num_attention_heads), control_params.get('attention_head_dim',self.config.attention_head_dim)
        inner_dim=num_attention_heads* attention_head_dim
        self.inner_dim=inner_dim
        
        # condition image patch embedding module
        control_pos_embed_input = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels + control_params.get('extra_conditioning_channels',0),
            embed_dim=inner_dim,
            pos_embed_max_size=control_params.get('pos_embed_max_size',self.config.pos_embed_max_size),
            pos_embed_type=control_params.get('pos_embed_type','sincos') if not self.use_rope else None,
        )
        # self.control_pos_embed_input = zero_module(control_pos_embed_input)
        self.control_pos_embed_input = control_pos_embed_input
        self.trainable_control_modules.update(dict(control_pos_embed_input=self.control_pos_embed_input))
        
        # target image patch embedding module
        if control_params.get('use_pos_embed',False):
            self.control_pos_embed = PatchEmbed(
                height=self.config.sample_size,
                width=self.config.sample_size,
                patch_size=self.config.patch_size,
                in_channels=self.config.in_channels,
                embed_dim=inner_dim,
                pos_embed_max_size=control_params.get('pos_embed_max_size',self.config.pos_embed_max_size),
                pos_embed_type=control_params.get('pos_embed_type','sincos') if not self.use_rope else None,
            )
            self.trainable_control_modules.update(dict(control_pos_embed=self.control_pos_embed))
        else:
            self.control_pos_embed = None
        
        # condition noise prediction time embedding module
        self.control_time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.trainable_control_modules.update(dict(control_time_text_embed=self.control_time_text_embed))
        
        self.control_condition_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.trainable_control_modules.update(dict(control_condition_embed=self.control_condition_embed))
        
        # condition attention block
        control_joint_attention_dim=control_params.get('joint_attention_dim',self.config.joint_attention_dim)
        self.control_context_embedder = nn.Linear(control_joint_attention_dim, inner_dim) # control_params['caption_projection_dim']
        self.trainable_control_modules.update(dict(control_context_embedder=self.control_context_embedder))
        
        if self.use_rope: 
            singletransblock=SingleRoPETransformerBlock
            jointtransblock=JointRoPETransformerBlock
        else:
            singletransblock=SD3SingleTransformerBlock
            jointtransblock=JointTransformerBlock
            
        if not self.use_encoder_hidden_states:
            self.control_transformer_blocks = nn.ModuleList(
                [
                    singletransblock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    )
                    for _ in range(self.control_blocks_num)
                ]
            )
        else:
            self.control_transformer_blocks = nn.ModuleList(
                [
                    jointtransblock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        context_pre_only=False,
                        qk_norm=control_params.get('qk_norm',self.config.qk_norm),
                        use_dual_attention=True if i in control_params.get('dual_attention_layers',self.config.dual_attention_layers) else False,
                    )
                    for i in range(self.control_blocks_num)
                ]
            )
        self.trainable_control_modules.update(dict(control_transformer_blocks=self.control_transformer_blocks))
        
        # control net attention block out add to mmdit
        self.controlnet_add_blocks = nn.ModuleList([])
        for _ in range(self.control_blocks_num):
            controlnet_block = nn.Linear(inner_dim, inner_dim)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_add_blocks.append(controlnet_block)
        self.trainable_control_modules.update(dict(controlnet_add_blocks=self.controlnet_add_blocks))
        
        if control_params.get('use_transformer_params',False):
            self.init_control_param()
        debug_print(f'Init controlnet block success......')
        
        if self.use_rope:
            self.rope_embed=FluxPosEmbed(theta=10000,axes_dim=(8, 28, 28))
        self.init_moe_block(inner_dim,num_attention_heads, attention_head_dim,control_params)

        self.cn_method=control_params.get("cn2base_method","add")
    
    def init_trainable_param(self):
        super().init_trainable_param()
        
        if self.cn_method=="CrossAttn":
            for name,param in self.transformer_blocks.named_parameters():
                if 'condition_' in name:
                    param.requires_grad_(True)
        
        debug_print('Configure the gradient derivation of transformer related parameters!')
    
    def init_control_param(self):
        if self.control_pos_embed is not None:
            self.control_pos_embed.load_state_dict(self.pos_embed.state_dict(),strict=False)
        self.control_pos_embed_input.load_state_dict(self.pos_embed.state_dict(),strict=False)
        
        self.control_time_text_embed.load_state_dict(self.time_text_embed.state_dict())
        self.control_condition_embed.load_state_dict(self.time_text_embed.state_dict())
        
        self.control_context_embedder.load_state_dict(self.context_embedder.state_dict())
            
        if self.use_encoder_hidden_states:
            self.control_transformer_blocks.load_state_dict(self.transformer_blocks.state_dict(), strict=False)

        debug_print('Initialize controlnet based on Transformer parameters....')
    
    def init_moe_block(self,inner_dim,num_attention_heads, attention_head_dim,control_params):
        self.expert_nums=control_params.get('expert_num',None) if control_params.get('expert_num',None) is not None else (self.condition_nums+1)*control_params.get('expert_num_each_condition',3)
        self.top_k=control_params.get('top_num',1)
        
        if self.use_rope: 
            singletransblock=SingleRoPETransformerBlock
            jointtransblock=JointRoPETransformerBlock
        else:
            singletransblock=SD3SingleTransformerBlock
            jointtransblock=JointTransformerBlock
        
        self.use_modulate=control_params.get('use_modulate',False)
        if self.use_modulate or self.use_rope:
            moe_init_denoise=nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(inner_dim,inner_dim),
                    nn.Linear(self.config.pooled_projection_dim,inner_dim)
                ]),
                nn.ModuleList([
                    nn.Linear(inner_dim,inner_dim),
                    nn.Linear(self.config.pooled_projection_dim,inner_dim)
                ])                
            ])
        else:
            moe_init_denoise=nn.ModuleList([
                singletransblock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    ), # for hidden states 
                singletransblock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    ) # for condition - hidden_states
            ])
        
        self.moe=MoE(inner_dim,moe_init_denoise,self.expert_nums,k=self.top_k)
        self.moe.moe_layer.experts.forward=self.expert_forward
        self.moe.forward=self.moe_forward
        self.num_local_experts=self.moe.num_local_experts
        self.trainable_control_modules.update(dict(moe=self.moe))
        
        self.use_shared_expert=control_params.get('use_shared_expert',False)
        if self.use_shared_expert:
            self.shared_expert = nn.ModuleList([
                jointtransblock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=False,
                    qk_norm=control_params.get('qk_norm',self.config.qk_norm),
                    use_dual_attention=False,
                ), # use condition info to constrain hidden states
                jointtransblock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=True,
                    qk_norm=control_params.get('qk_norm',self.config.qk_norm),
                    use_dual_attention=True,
                ) # use encoder hidden states to constrain hidden states and condition hidden states
            ])
            self.trainable_control_modules.update(dict(shared_expert=self.shared_expert))
        
    def expert_forward(
        self,
        hidden_states,
        condition_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
        condition_temb: torch.Tensor = None,
        **kwargs
    ):
        expert_hidden_out: List[torch.Tensor] = []
        expert_condition_hidden_out: List[torch.Tensor] = []
        
        hidden_chunks = hidden_states.chunk(self.num_local_experts, dim=1)
        condition_chunks=condition_hidden_states.chunk(self.num_local_experts,dim=1)
        temb_chunks=temb.chunk(self.num_local_experts,dim=1)
        condition_temb_chunks=condition_temb.chunk(self.num_local_experts,dim=1)
        
        condition_pooled_embeds=kwargs.get('condition_pooled_projections',None)
        pooled_embeds=kwargs.get('pooled_projections',None)
        
        if self.use_modulate: assert condition_pooled_embeds is not None and pooled_embeds is not None, ValueError('Check MoE Forward input, condition pooled projection or pooled projection is None!')
        
        condition_pooled_chunks=condition_pooled_embeds.chunk(self.num_local_experts,dim=1)
        pooled_chunks=pooled_embeds.chunk(self.num_local_experts,dim=1)
        
        for expert_idx,(hidden_chunk,condition_chunk,temb_chunk,condition_temb_chunk, expert) in enumerate(zip(hidden_chunks,condition_chunks,temb_chunks,condition_temb_chunks,self.moe.moe_layer.experts.deepspeed_experts)):
            hidden_chunk,condition_chunk,temb_chunk,condition_temb_chunk=hidden_chunk.squeeze(1),condition_chunk.squeeze(1),temb_chunk.squeeze(1),condition_temb_chunk.squeeze(1)
            
            if self.use_modulate or self.use_rope:
                condition_pooled_chunk=condition_pooled_chunks[expert_idx].squeeze(1)
                pooled_chunk=pooled_chunks[expert_idx].squeeze(1)
                
                condition_modulate,hidden_modulate=expert[0],expert[1]
                condition_chunk=modulated_flatten(condition_chunk,condition_modulate[0].weight,condition_modulate[1](condition_pooled_chunk))+condition_modulate[0].bias.unsqueeze(0)
                
                hidden_chunk=modulated_flatten(hidden_chunk+condition_chunk,hidden_modulate[0].weight,hidden_modulate[1](pooled_chunk))+hidden_modulate[0].bias.unsqueeze(0)
            else:
                hidden_chunk=expert[0](hidden_chunk,temb_chunk)
                condition_chunk=expert[1](condition_chunk,condition_temb_chunk)
            
            expert_hidden_out.append(hidden_chunk)
            expert_condition_hidden_out.append(condition_chunk)

        return (torch.stack(expert_hidden_out, dim=1),torch.stack(expert_condition_hidden_out, dim=1))
    
    def moe_forward(self,
                    hidden_states: torch.Tensor,
                    condition_hidden_states: torch.Tensor,
                    used_token: Optional[torch.Tensor] = None,
                    **kwargs):
        joint_attention_kwargs=kwargs.pop('joint_attention_kwargs',dict())
        if self.use_rope:
            assert len(joint_attention_kwargs.keys())!=0, ValueError('Please give joint_attention_kwargs info when use rope!')
        else:
            new_joint_attention_kwargs=dict()
        expert_output=self.moe.moe_layer(choice_expert_input=hidden_states+condition_hidden_states,hidden_states=hidden_states,condition_hidden_states=condition_hidden_states,used_token=used_token,**kwargs)
        
        expert_hidden_states,expert_condition_states=expert_output
        if self.use_shared_expert:
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=joint_attention_kwargs['img_ids'],encoder_hd_ids=joint_attention_kwargs['condition_ids'],rope_embed=self.rope_embed)
            condition_states,hidden_states=self.shared_expert[0](hidden_states,condition_hidden_states,temb=kwargs['condition_temb'],joint_attention_kwargs=new_joint_attention_kwargs)
            
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=torch.cat([joint_attention_kwargs['img_ids'],joint_attention_kwargs['condition_ids']],dim=0),encoder_hd_ids=joint_attention_kwargs['prompt_ids'],rope_embed=self.rope_embed)
            hidden_condition_states=torch.cat([hidden_states,condition_states],dim=1)
            _,hidden_condition_states=self.shared_expert[1](hidden_condition_states,kwargs['encoder_hidden_states'],temb=kwargs['temb'],joint_attention_kwargs=new_joint_attention_kwargs)
            
            hidden_states,condition_states=hidden_condition_states[:,:hidden_states.shape[1],:],hidden_condition_states[:,hidden_states.shape[1]:,:]
            
            expert_output=(hidden_states+expert_hidden_states,condition_states+expert_condition_states)
            
        return expert_output, self.moe.moe_layer.l_aux, self.moe.moe_layer.exp_counts
    
    def control_forward(
        self,                   
        hidden_states,
        condition_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        conditioning_scale: float = 1.0,
        condition_pooled_projections: torch.LongTensor = None,
        **kwargs):
        joint_attention_kwargs=dict()
        
        # prepare control inputs
        if self.control_pos_embed is not None:
            control_hidden_states = self.control_pos_embed(hidden_states)
        else:
            control_hidden_states=self.pos_embed(hidden_states)    
        
        condition_hidden_states=self.control_pos_embed_input(condition_hidden_states)
        
        if self.use_pooled_prompt_embeds:
            control_pooled_projections=pooled_projections
        else:
            control_pooled_projections=torch.zeros_like(pooled_projections)
        
        control_temb=self.control_time_text_embed(timestep,control_pooled_projections)
        condition_temb=self.control_condition_embed(timestep,condition_pooled_projections)
        
        control_encoder_hidden_states=self.control_context_embedder(encoder_hidden_states)

        if self.use_rope:
            joint_attention_kwargs=dict(hd_ids=kwargs['img_ids'],encoder_hd_ids=kwargs['prompt_ids'],rope_embed=self.rope_embed)
        control_encoder_hidden_states,control_hidden_states = self.preprocess_block[0](control_hidden_states,control_encoder_hidden_states,control_temb,joint_attention_kwargs=joint_attention_kwargs)
        
        if self.use_rope:
            joint_attention_kwargs=dict(hd_ids=torch.cat((kwargs['img_ids'], kwargs['prompt_ids']), dim=0),encoder_hd_ids=kwargs['condition_ids'],rope_embed=self.rope_embed)
        condition_hidden_states,control_hidden_encoder_states=self.preprocess_block[1](torch.cat([control_hidden_states,control_encoder_hidden_states],dim=1),condition_hidden_states,condition_temb,joint_attention_kwargs=joint_attention_kwargs)
        
        control_hidden_states,control_encoder_hidden_states=control_hidden_encoder_states[:,:control_hidden_states.shape[1],:],control_hidden_encoder_states[:,control_hidden_states.shape[1]:,:]
        
        if self.use_rope:
            joint_attention_kwargs=dict(img_ids=kwargs['img_ids'],prompt_ids=kwargs['prompt_ids'],condition_ids=kwargs['condition_ids'],rope_embed=self.rope_embed)
        moe_output,moe_loss,exp_count=self.moe(hidden_states=control_hidden_states,condition_hidden_states=condition_hidden_states,encoder_hidden_states=control_encoder_hidden_states,temb=control_temb,condition_temb=condition_temb,condition_pooled_projections=condition_pooled_projections,pooled_projections=pooled_projections,joint_attention_kwargs=joint_attention_kwargs)
        
        expert_hidden_states,expert_condition_hidden_states=moe_output
        # -------
        hidden_states=expert_condition_hidden_states+expert_hidden_states

        controlnet_block_res_samples=[]
        for index_block,block in enumerate(self.control_transformer_blocks):
            if self.use_encoder_hidden_states:
                if self.use_rope:
                    joint_attention_kwargs=dict(hd_ids=kwargs['img_ids'],encoder_hd_ids=kwargs['prompt_ids'],rope_embed=self.rope_embed)
                control_encoder_hidden_states,hidden_states = block(hidden_states,control_encoder_hidden_states,control_temb,joint_attention_kwargs=joint_attention_kwargs)
            else:
                if self.use_rope:
                    joint_attention_kwargs=dict(hd_ids=kwargs['img_ids'],rope_embed=self.rope_embed)
                hidden_states = block(hidden_states, control_temb,joint_attention_kwargs=joint_attention_kwargs)      

            controlnet_block=self.controlnet_add_blocks[index_block]
            if self.use_modulate and hasattr(self,'add_cn'):
                modulated_bloack=self.add_cn[index_block]
                modulated_hidden_states=modulated_flatten(hidden_states,modulated_bloack[0].weight,modulated_bloack[1](condition_pooled_projections))+modulated_bloack[0].bias.unsqueeze(0)
                modulated_hidden_states=hidden_states
            else:
                modulated_hidden_states=hidden_states
            block_res_sample = controlnet_block(modulated_hidden_states)
            controlnet_block_res_samples.append(block_res_sample*conditioning_scale)
            
        return controlnet_block_res_samples,moe_loss,exp_count
    
    def base_forward(
        self,
        hidden_states,
        encoder_hidden_states: torch.FloatTensor = None,
        control_hidden_states: List[torch.Tensor] = None,
        temb: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None):

        joint_attention_kwargs=joint_attention_kwargs or dict()
        
        each_block_hidden_states,each_block_context_hidden_states=[],[]

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            
            if control_hidden_states is not None:
                interval_control = len(self.transformer_blocks) / len(control_hidden_states)
                if self.cn_method == "CrossAttn":
                    joint_attention_kwargs['condition_hidden_states']=control_hidden_states[int(index_block / interval_control)]
                elif self.cn_method == "add":
                    hidden_states = hidden_states + control_hidden_states[int(index_block / interval_control)]
            
            each_block_hidden_states.append(hidden_states)
            each_block_context_hidden_states.append(encoder_hidden_states)
        
        return each_block_hidden_states,each_block_context_hidden_states
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        condition_hidden_states: torch.FloatTensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        skip_layers: Optional[List[int]] = None,
        **kwargs
    ):
        add_losses,add_outputs=dict(),dict()
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]
        
        if self.use_rope:
            img_ids=prepare_latent_image_ids(height//2,width//2,hidden_states.device,hidden_states.dtype)
            prompt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            condition_ids=prepare_latent_image_ids(condition_hidden_states.shape[2]//2,condition_hidden_states.shape[3]//2,device=condition_hidden_states.device,dtype=condition_hidden_states.dtype)
        else:
            img_ids,prompt_ids,condition_ids=None,None,None
            
        cn_blocks_hidden_states,moe_loss,exp_count=self.control_forward(hidden_states=hidden_states,
                                                condition_hidden_states=condition_hidden_states,
                                                encoder_hidden_states=encoder_hidden_states,
                                                pooled_projections=pooled_projections,
                                                timestep=timestep,
                                                conditioning_scale=conditioning_scale,
                                                joint_attention_kwargs=joint_attention_kwargs,
                                                condition_pooled_projections=condition_pooled_projections,
                                                img_ids=img_ids,prompt_ids=prompt_ids,condition_ids=condition_ids
                                                )
        add_losses.update(dict(moe_loss=moe_loss*0.1))
        add_outputs.update(dict(expert_counts=exp_count))
        
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        blocks_hidden_states,_=self.base_forward(hidden_states,
                                               encoder_hidden_states=encoder_hidden_states,
                                               control_hidden_states=cn_blocks_hidden_states,
                                               temb=temb,
                                               joint_attention_kwargs=joint_attention_kwargs)

        hidden_states = self.norm_out(blocks_hidden_states[-1], temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        return output,add_losses,add_outputs

class UniGenSD3(UniGenBase):
    def init_control_block(self, control_params=None):
        super().init_control_block(control_params)
        self.control_context_embedder = nn.Linear(self.inner_dim, self.inner_dim)
        self.trainable_control_modules.update(dict(control_context_embedder=self.control_context_embedder))    

        assert self.use_encoder_hidden_states, ValueError('please use joint transformer block to enhance condition hidden states')

    def preprocess_moe_forward(
        self,
        hidden_states,
        condition_hidden_states,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.LongTensor = None,
        timestep: torch.LongTensor = None,
        **kwargs
    ):  
        joint_attention_kwargs=dict()
        
        condition_hidden_states=self.control_pos_embed_input(condition_hidden_states)
        
        if self.use_pooled_prompt_embeds:
            control_pooled_projections=pooled_projections
        else:
            control_pooled_projections=torch.zeros_like(pooled_projections)
            
        control_temb=self.control_time_text_embed(timestep,control_pooled_projections)
        condition_temb=self.control_condition_embed(timestep,condition_pooled_projections)
        
        control_encoder_hidden_states=self.control_context_embedder(encoder_hidden_states)
        
        if self.use_rope:
            joint_attention_kwargs=dict(img_ids=kwargs['img_ids'],prompt_ids=kwargs['prompt_ids'],condition_ids=kwargs['condition_ids'],rope_embed=self.rope_embed)
        moe_output,moe_loss,exp_count=self.moe(hidden_states=hidden_states,condition_hidden_states=condition_hidden_states,encoder_hidden_states=control_encoder_hidden_states,temb=control_temb,condition_temb=condition_temb,condition_pooled_projections=condition_pooled_projections,pooled_projections=pooled_projections,joint_attention_kwargs=joint_attention_kwargs)
        
        expert_hidden_states,expert_condition_hidden_states=moe_output
        
        return dict(
                    condition_hidden_states=condition_hidden_states,
                    expert_hidden_states=expert_hidden_states,
                    expert_condition_hidden_states=expert_condition_hidden_states,
                    control_encoder_hidden_states=control_encoder_hidden_states,
                    control_temb=control_temb,
                    condition_temb=condition_temb,
                    exp_count=exp_count,
                    moe_loss=moe_loss,
                    )
    
    def control_forward(
        self,                   
        hidden_states,
        condition_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        conditioning_scale: float = 1.0,
        condition_pooled_projections: torch.LongTensor = None,
        base_block_idx: int = None,
        **kwargs):
        joint_attention_kwargs=dict()
        
        assert base_block_idx is not None, ValueError('Please provide cn_block_idx to select control net transformer block!')
        interval_control = len(self.transformer_blocks) / len(self.control_transformer_blocks)
        cn_block_idx=int(base_block_idx / interval_control)
        
        block=self.control_transformer_blocks[cn_block_idx]

        moe_output=kwargs.pop('moe_output',None)
        if base_block_idx==0:
            moe_output=self.preprocess_moe_forward(hidden_states,condition_hidden_states,encoder_hidden_states,
                                            pooled_projections,condition_pooled_projections,timestep,**kwargs)
        
            hidden_states=moe_output['expert_hidden_states']+moe_output['expert_condition_hidden_states']
        else:
            assert moe_output is not None, ValueError('Check moe output parameter info!')
            
        encoder_hidden_states=moe_output['control_encoder_hidden_states']
        temb=moe_output['condition_temb']
        
        if self.use_encoder_hidden_states:
            if self.use_rope:
                joint_attention_kwargs=dict(hd_ids=kwargs['img_ids'],encoder_hd_ids=kwargs['prompt_ids'],rope_embed=self.rope_embed)
            encoder_hidden_states,hidden_states = block(hidden_states,encoder_hidden_states,temb,joint_attention_kwargs=joint_attention_kwargs)
        else:
            if self.use_rope:
                joint_attention_kwargs=dict(hd_ids=kwargs['img_ids'],rope_embed=self.rope_embed)
            hidden_states = block(hidden_states, temb,joint_attention_kwargs=joint_attention_kwargs)      

        return self.controlnet_add_blocks[cn_block_idx](hidden_states)*conditioning_scale,hidden_states,moe_output
    
    def base_forward(
        self,
        hidden_states,
        condition_hidden_states,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.Tensor = None,
        timestep: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        temb: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs):
        joint_attention_kwargs = joint_attention_kwargs or dict()

        moe_output=None
        each_block_hidden_states,each_block_context_hidden_states=[],[]

        for index_block, block in enumerate(self.transformer_blocks):
            
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            
            zero_condition_hidden_states,condition_hidden_states,moe_output=self.control_forward(hidden_states,condition_hidden_states,encoder_hidden_states,
                                                  pooled_projections,timestep,conditioning_scale,condition_pooled_projections,
                                                  index_block,moe_output=moe_output,**kwargs)
            
            if self.cn_method == "CrossAttn":
                joint_attention_kwargs['condition_hidden_states']=condition_hidden_states
            
            hidden_states=hidden_states+zero_condition_hidden_states
            
            # each_block_hidden_states.append(hidden_states)
            # each_block_context_hidden_states.append(encoder_hidden_states)
        
        return dict(blocks_hidden_states=hidden_states,
                    block_ctx_hidden_states=encoder_hidden_states,
                    moe_loss=moe_output['moe_loss'],
                    exp_count=moe_output['exp_count']
                )
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        condition_hidden_states: torch.FloatTensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        skip_layers: Optional[List[int]] = None,
        **kwargs
    ):
        add_losses,add_outputs=dict(),dict()
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]
        
        if self.use_rope:
            img_ids=prepare_latent_image_ids(height//2,width//2,hidden_states.device,hidden_states.dtype)
            prompt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            condition_ids=prepare_latent_image_ids(condition_hidden_states.shape[2]//2,condition_hidden_states.shape[3]//2,device=condition_hidden_states.device,dtype=condition_hidden_states.dtype)
        else:
            img_ids,prompt_ids,condition_ids=None,None,None

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        control_encoder_hidden_states=encoder_hidden_states
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        forward_res=self.base_forward(hidden_states=hidden_states,
                                      condition_hidden_states=condition_hidden_states,
                                      encoder_hidden_states=encoder_hidden_states,
                                      pooled_projections=pooled_projections,
                                      condition_pooled_projections=condition_pooled_projections,
                                      timestep=timestep,
                                      conditioning_scale=conditioning_scale,
                                      temb=temb,
                                      joint_attention_kwargs=joint_attention_kwargs,
                                      control_encoder_hidden_states=control_encoder_hidden_states,
                                      img_ids=img_ids,prompt_ids=prompt_ids,condition_ids=condition_ids
                                    )

        add_losses.update(dict(moe_loss=forward_res['moe_loss']*0.1))
        add_outputs.update(dict(expert_counts=forward_res['exp_count']))
        
        hidden_states = self.norm_out(forward_res['blocks_hidden_states'], temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        return output,add_losses,add_outputs

class UniGenFlux(FluxTransformer2DModel):
    def init_condition_block(self,condition_nums=1,**kwargs): 
        self.condition_nums=condition_nums   
        self.init_control_block(kwargs.get('control_params',None))
    
    def init_control_block(self, control_params=None):
        assert control_params is not None, ValueError('Please provice control net model parameter')
        self.trainable_control_modules=dict()
        self.use_pooled_prompt_embeds=control_params.get('use_pooled_prompt_embeds',True)
        self.use_encoder_hidden_states=control_params.get('use_encoder_hidden_states',True)
        # self.control_blocks_num=control_params.get('num_layers',self.config.num_layers)
        self.config.hidden_size=self.inner_dim
        self.use_rope=control_params.get('use_rope',False)
        
        # condition image patch embedding module
        self.control_pos_embed_input = copy.deepcopy(self.pos_embed)
        self.trainable_control_modules.update(dict(control_pos_embed_input=self.control_pos_embed_input))
        
        # condition noise prediction time embedding module
        self.control_time_text_embed = copy.deepcopy(self.time_text_embed)
        self.trainable_control_modules.update(dict(control_time_text_embed=self.control_time_text_embed))
        
        self.control_condition_embed = copy.deepcopy(self.time_text_embed)
        self.trainable_control_modules.update(dict(control_condition_embed=self.control_condition_embed))
        
        # condition attention block
        self.control_context_embedder = nn.Linear(self.inner_dim,self.inner_dim) # control_params['caption_projection_dim']
        self.trainable_control_modules.update(dict(control_context_embedder=self.control_context_embedder))
        
        self.control_x_embedder=copy.deepcopy(self.x_embedder)
        self.trainable_control_modules.update(dict(control_x_embedder=self.control_x_embedder))
        
        self.cn_joint_layers,self.cn_single_joint_layers=len(self.transformer_blocks)//control_params.get('single_control_dev',2),len(self.single_transformer_blocks)//control_params.get('single_control_dev',2)
        self.control_joint_trans_blocks=nn.ModuleList([
            FluxJointRoPETransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                ) for _ in range(self.cn_joint_layers) 
        ])
        self.trainable_control_modules.update(dict(control_joint_trans_blocks=self.control_joint_trans_blocks))

        # control net attention block out add to mmdit
        self.controlnet_add_joint_blocks, self.controlnet_add_single_blocks = nn.ModuleList([]), nn.ModuleList([])
        for _ in range(self.cn_joint_layers):
            self.controlnet_add_joint_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))
        self.trainable_control_modules.update(dict(controlnet_add_joint_blocks=self.controlnet_add_joint_blocks))
    
        if control_params.get('use_single_trans_blocks',True):
            self.single_block_control_method=control_params.get("single_block_control_method","overall_add") # overall_add: hidden_states + controlnet_out; single_add: hidd_states[:, encoder_len:, ...] + controlnet_out[:, encoder_len:, ...]
            self.control_single_trans_blocks=nn.ModuleList([
                FluxSingleRoPETransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=self.config.num_attention_heads,
                        attention_head_dim=self.config.attention_head_dim,
                    ) for _ in range(self.cn_single_joint_layers) 
            ])
            self.trainable_control_modules.update(dict(control_single_trans_blocks=self.control_single_trans_blocks))
            
            for _ in range(self.cn_single_joint_layers):
                self.controlnet_add_single_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))
            self.trainable_control_modules.update(dict(controlnet_add_single_blocks=self.controlnet_add_single_blocks))

            debug_print(f'using single transformer blocks, single_control_dev: {control_params.get("single_control_dev",2)}, control_single_blocks: {self.cn_single_joint_layers}, control method: {self.single_block_control_method}')

        if control_params.get('use_transformer_params',False):
            self.init_control_param()
            debug_print(f'Init controlnet block success......')
        
        self.init_moe_block(self.inner_dim,self.config.num_attention_heads, self.config.attention_head_dim,control_params)

        self.cn_method=control_params.get("cn2base_method","add")
    
    def init_trainable_param(self):
        for name,module in self.trainable_control_modules.items():
            module.requires_grad_(True)
    
    def init_control_param(self):
        self.control_pos_embed_input.load_state_dict(self.pos_embed.state_dict(),strict=False)
        
        self.control_time_text_embed.load_state_dict(self.time_text_embed.state_dict())
        self.control_condition_embed.load_state_dict(self.time_text_embed.state_dict())
        
        # self.control_context_embedder.load_state_dict(self.context_embedder.state_dict())

        self.control_x_embedder.load_state_dict(self.control_x_embedder.state_dict())

        self.control_joint_trans_blocks.load_state_dict(self.transformer_blocks.state_dict(),strict=False)
        if hasattr(self,'control_single_trans_blocks'):
            self.control_single_trans_blocks.load_state_dict(self.single_transformer_blocks.state_dict(),strict=False)
            
        # self.control_pos_embed_input = zero_module(self.control_pos_embed_input)
        debug_print('Initialize controlnet based on Transformer parameters....')
    
    def init_moe_block(self,inner_dim,num_attention_heads, attention_head_dim,control_params):
        self.expert_nums=control_params.get('expert_num',None) if control_params.get('expert_num',None) is not None else (self.condition_nums+1)*control_params.get('expert_num_each_condition',3)
        self.top_k=control_params.get('top_num',1)
        
        if self.use_rope: 
            singletransblock=FluxSingleRoPETransformerBlock
            jointtransblock=FluxJointRoPETransformerBlock
        else:
            singletransblock=FluxSingleTransformerBlock
            jointtransblock=FluxTransformerBlock
            
        # self.preprocess_block=nn.ModuleList([
        #     jointtransblock(
        #                 dim=inner_dim,
        #                 num_attention_heads=num_attention_heads,
        #                 attention_head_dim=attention_head_dim,
        #             ),
        #     jointtransblock(
        #                 dim=inner_dim,
        #                 num_attention_heads=num_attention_heads,
        #                 attention_head_dim=attention_head_dim,
        #             )
        # ])
        # self.trainable_control_modules.update(dict(preprocess_block=self.preprocess_block))
        
        self.use_modulate=control_params.get('use_modulate',False)
        if self.use_modulate or self.use_rope:
            moe_init_denoise=nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(inner_dim,inner_dim),
                    nn.Linear(self.config.pooled_projection_dim,inner_dim)
                ]),
                nn.ModuleList([
                    nn.Linear(inner_dim,inner_dim),
                    nn.Linear(self.config.pooled_projection_dim,inner_dim)
                ])                
            ])
        else:
            moe_init_denoise=nn.ModuleList([
                singletransblock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    ), # for hidden states 
                singletransblock(
                        dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    ) # for condition - hidden_states
            ])
        
        self.moe=MoE(inner_dim,moe_init_denoise,self.expert_nums,k=self.top_k)
        self.moe.moe_layer.experts.forward=self.expert_forward
        self.moe.forward=self.moe_forward
        self.num_local_experts=self.moe.num_local_experts
        self.trainable_control_modules.update(dict(moe=self.moe))
        
        self.use_shared_expert=control_params.get('use_shared_expert',False)
        if self.use_shared_expert:
            # V1
            """
            self.shared_expert = jointtransblock(
                                    dim=inner_dim,
                                    num_attention_heads=num_attention_heads,
                                    attention_head_dim=attention_head_dim,
                                    context_pre_only=False,
                                    qk_norm=control_params.get('qk_norm',self.config.qk_norm),
                                    use_dual_attention=False,
                                )
            # coefficient is used for weighted sum of the output of expert and mlp
            # self.coefficient = nn.Linear(inner_dim, 2)
            """
            # V2
            self.shared_expert = nn.ModuleList([
                jointtransblock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                ), # use condition info to constrain hidden states
                jointtransblock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                ) # use encoder hidden states to constrain hidden states and condition hidden states
            ])
            self.trainable_control_modules.update(dict(shared_expert=self.shared_expert))
        
        self.use_consis_module=control_params.get('use_consis_module',False)
        if self.use_consis_module:
            """
            # V1
            self.consis_module=nn.ModuleList([
                singletransblock(
                    inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim
                ),
                singletransblock(
                    inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim
                )
            ])
            """
            # V2
            self.consis_module=nn.ModuleList([
                jointtransblock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                ),
                jointtransblock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
            ])
            self.trainable_control_modules.update(dict(consis_module=self.consis_module))
        
    def expert_forward(
        self,
        hidden_states,
        condition_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
        condition_temb: torch.Tensor = None,
        **kwargs
    ):
        expert_hidden_out: List[torch.Tensor] = []
        expert_condition_hidden_out: List[torch.Tensor] = []
        
        hidden_chunks = hidden_states.chunk(self.num_local_experts, dim=1)
        condition_chunks=condition_hidden_states.chunk(self.num_local_experts,dim=1)
        temb_chunks=temb.chunk(self.num_local_experts,dim=1)
        condition_temb_chunks=condition_temb.chunk(self.num_local_experts,dim=1)
        
        condition_pooled_embeds=kwargs.get('condition_pooled_projections',None)
        pooled_embeds=kwargs.get('pooled_projections',None)
        
        if self.use_modulate: assert condition_pooled_embeds is not None and pooled_embeds is not None, ValueError('Check MoE Forward input, condition pooled projection or pooled projection is None!')
        
        condition_pooled_chunks=condition_pooled_embeds.chunk(self.num_local_experts,dim=1)
        pooled_chunks=pooled_embeds.chunk(self.num_local_experts,dim=1)
        
        for expert_idx,(hidden_chunk,condition_chunk,temb_chunk,condition_temb_chunk, expert) in enumerate(zip(hidden_chunks,condition_chunks,temb_chunks,condition_temb_chunks,self.moe.moe_layer.experts.deepspeed_experts)):
            hidden_chunk,condition_chunk,temb_chunk,condition_temb_chunk=hidden_chunk.squeeze(1),condition_chunk.squeeze(1),temb_chunk.squeeze(1),condition_temb_chunk.squeeze(1)
            
            if self.use_modulate or self.use_rope:
                condition_pooled_chunk=condition_pooled_chunks[expert_idx].squeeze(1)
                pooled_chunk=pooled_chunks[expert_idx].squeeze(1)
                
                condition_modulate,hidden_modulate=expert[0],expert[1]
                condition_chunk=modulated_flatten(condition_chunk,condition_modulate[0].weight,condition_modulate[1](condition_pooled_chunk))+condition_modulate[0].bias.unsqueeze(0)
                
                hidden_chunk=modulated_flatten(hidden_chunk+condition_chunk,hidden_modulate[0].weight,hidden_modulate[1](pooled_chunk))+hidden_modulate[0].bias.unsqueeze(0)
            else:
                hidden_chunk=expert[0](hidden_chunk,temb_chunk)
                condition_chunk=expert[1](condition_chunk,condition_temb_chunk)
            
            expert_hidden_out.append(hidden_chunk)
            expert_condition_hidden_out.append(condition_chunk)

        return (torch.stack(expert_hidden_out, dim=1),torch.stack(expert_condition_hidden_out, dim=1))
    
    def moe_forward(self,
                    hidden_states: torch.Tensor,
                    condition_hidden_states: torch.Tensor,
                    used_token: Optional[torch.Tensor] = None,
                    **kwargs):
        joint_attention_kwargs=kwargs.pop('joint_attention_kwargs',dict())
        if self.use_rope:
            assert len(joint_attention_kwargs.keys())!=0, ValueError('Please give joint_attention_kwargs info when use rope!')
        else:
            new_joint_attention_kwargs=dict()
        expert_output=self.moe.moe_layer(choice_expert_input=hidden_states+condition_hidden_states,hidden_states=hidden_states,condition_hidden_states=condition_hidden_states,used_token=used_token,**kwargs)
        
        expert_hidden_states,expert_condition_states=expert_output
        if self.use_consis_module:
            """
            # V1
            consis_expert_hidden=self.consis_module[0](expert_hidden_states,temb=kwargs['temb'])
            consis_expert_condition_hidden=self.consis_module[1](expert_condition_states,temb=kwargs['condition_temb'])
            
            expert_hidden_states=expert_hidden_states+consis_expert_hidden
            expert_condition_states=expert_condition_states+consis_expert_condition_hidden
            """
            #V2
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=joint_attention_kwargs['condition_ids'],encoder_hd_ids=joint_attention_kwargs['condition_ids'],rope_embed=self.pos_embed)
            _,consis_expert_condition_hidden=self.consis_module[0](expert_condition_states,condition_hidden_states,temb=kwargs['condition_temb'],joint_attention_kwargs=new_joint_attention_kwargs)
            
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=torch.cat([joint_attention_kwargs['img_ids'],joint_attention_kwargs['condition_ids']],dim=0),encoder_hd_ids=joint_attention_kwargs['img_ids'],rope_embed=self.pos_embed)
            _,consis_expert_hidden_condition_states=self.consis_module[0](torch.cat([expert_hidden_states,consis_expert_condition_hidden],dim=1),hidden_states,temb=kwargs['temb'],joint_attention_kwargs=new_joint_attention_kwargs)
            
            consis_expert_hidden,consis_expert_condition_hidden=consis_expert_hidden_condition_states[:,:expert_hidden_states.shape[1],:],consis_expert_hidden_condition_states[:,expert_hidden_states.shape[1]:,:]
            
            expert_hidden_states=expert_hidden_states+consis_expert_hidden
            expert_condition_states=expert_condition_states+consis_expert_condition_hidden
            
        if self.use_shared_expert:
            # V1
            """
            hidden_encoder_states=torch.cat([hidden_states,kwargs['encoder_hidden_states']],dim=1)
            condition_states,hidden_encoder_states = self.shared_expert(hidden_encoder_states,kwargs['condition_hidden_states'],temb=kwargs['temb']+kwargs['condition_temb'])
            hidden_states,encoder_hidden_states=hidden_encoder_states[:,:expert_hidden_states.shape[1],:],hidden_encoder_states[:,expert_hidden_states.shape[1]:,:]
            """
            # V2
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=joint_attention_kwargs['img_ids'],encoder_hd_ids=joint_attention_kwargs['condition_ids'],rope_embed=self.pos_embed)
            condition_states,hidden_states=self.shared_expert[0](hidden_states,condition_hidden_states,temb=kwargs['condition_temb'],joint_attention_kwargs=new_joint_attention_kwargs)
            
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=torch.cat([joint_attention_kwargs['img_ids'],joint_attention_kwargs['condition_ids']],dim=0),encoder_hd_ids=joint_attention_kwargs['prompt_ids'],rope_embed=self.pos_embed)
            hidden_condition_states=torch.cat([hidden_states,condition_states],dim=1)
            _,hidden_condition_states=self.shared_expert[1](hidden_condition_states,kwargs['encoder_hidden_states'],temb=kwargs['temb'],joint_attention_kwargs=new_joint_attention_kwargs)
            
            hidden_states,condition_states=hidden_condition_states[:,:hidden_states.shape[1],:],hidden_condition_states[:,hidden_states.shape[1]:,:]
            
            expert_output=(hidden_states+expert_hidden_states,condition_states+expert_condition_states)
            
        return expert_output, self.moe.moe_layer.l_aux, self.moe.moe_layer.exp_counts
    
    def preprocess_moe_forward(
        self,
        hidden_states,
        condition_hidden_states,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.LongTensor = None,
        timestep: torch.LongTensor = None,
        **kwargs
    ):  
        joint_attention_kwargs=dict()
        
        condition_hidden_states=self.control_x_embedder(condition_hidden_states)
        
        if self.use_pooled_prompt_embeds:
            control_pooled_projections=pooled_projections
        else:
            control_pooled_projections=torch.zeros_like(pooled_projections)
        
        
        control_temb=self.control_time_text_embed(timestep,control_pooled_projections) if kwargs['guidance'] is None else self.control_time_text_embed(timestep,kwargs['guidance'],control_pooled_projections)
        condition_temb=self.control_condition_embed(timestep,condition_pooled_projections) if kwargs['guidance'] is None else self.control_condition_embed(timestep,kwargs['guidance'],condition_pooled_projections)
        
        control_encoder_hidden_states=self.control_context_embedder(encoder_hidden_states)
        
        if self.use_rope:
            joint_attention_kwargs=dict(img_ids=kwargs['img_ids'],prompt_ids=kwargs['prompt_ids'],condition_ids=kwargs['condition_ids'],rope_embed=self.control_pos_embed_input)
        moe_output,moe_loss,exp_count=self.moe(hidden_states=hidden_states,condition_hidden_states=condition_hidden_states,encoder_hidden_states=control_encoder_hidden_states,temb=control_temb,condition_temb=condition_temb,condition_pooled_projections=condition_pooled_projections,pooled_projections=pooled_projections,joint_attention_kwargs=joint_attention_kwargs)
        
        expert_hidden_states,expert_condition_hidden_states=moe_output
        
        return dict(
                    condition_hidden_states=condition_hidden_states,
                    expert_hidden_states=expert_hidden_states,
                    expert_condition_hidden_states=expert_condition_hidden_states,
                    control_encoder_hidden_states=control_encoder_hidden_states,
                    control_temb=control_temb,
                    condition_temb=condition_temb,
                    exp_count=exp_count,
                    moe_loss=moe_loss,
                    )
    
    def control_forward(
        self,                   
        hidden_states,
        condition_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        conditioning_scale: float = 1.0,
        condition_pooled_projections: torch.LongTensor = None,
        block: nn.Module = None,
        cn_block: nn.Module= None,
        **kwargs):
        joint_attention_kwargs=dict()

        moe_output=kwargs.pop('moe_output',None)
        if moe_output is None:
            moe_output=self.preprocess_moe_forward(hidden_states,condition_hidden_states,encoder_hidden_states,
                                            pooled_projections,condition_pooled_projections,timestep,**kwargs)
        
            hidden_states=moe_output['expert_hidden_states']+moe_output['expert_condition_hidden_states']
            
        encoder_hidden_states=moe_output['control_encoder_hidden_states']
        temb=moe_output['condition_temb']
        
        if "single_hd" not in moe_output:
            if self.use_rope:
                joint_attention_kwargs=dict(hd_ids=kwargs['img_ids'],encoder_hd_ids=kwargs['prompt_ids'],rope_embed=self.control_pos_embed_input)
            encoder_hidden_states,hidden_states = block(hidden_states,encoder_hidden_states,temb,joint_attention_kwargs=joint_attention_kwargs)
        else:
            if self.use_rope:
                joint_attention_kwargs=dict(hd_ids=torch.cat((kwargs['prompt_ids'],kwargs['img_ids']),dim=0),rope_embed=self.control_pos_embed_input)
            
            hidden_states = block(hidden_states, temb,joint_attention_kwargs=joint_attention_kwargs)      

        return cn_block(hidden_states)*conditioning_scale,hidden_states,moe_output
    
    def base_forward(
        self,
        hidden_states,
        condition_hidden_states,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.Tensor = None,
        timestep: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        temb: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        image_rotary_emb: torch.FloatTensor = None,
        **kwargs):
        joint_attention_kwargs = joint_attention_kwargs or dict()

        moe_output=None
        each_block_hidden_states,each_block_context_hidden_states=[],[]

        for index_block, block in enumerate(self.transformer_blocks):
            
            interval_control = len(self.transformer_blocks) / len(self.control_joint_trans_blocks)
            cn_block_idx=int(index_block / interval_control)
            
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            
            zero_condition_hidden_states,condition_hidden_states,moe_output=self.control_forward(hidden_states,condition_hidden_states,encoder_hidden_states,
                                                  pooled_projections,timestep,conditioning_scale,condition_pooled_projections,
                                                  self.control_joint_trans_blocks[cn_block_idx],self.controlnet_add_joint_blocks[cn_block_idx],moe_output=moe_output,**kwargs)
            
            hidden_states=hidden_states+zero_condition_hidden_states
            
            # each_block_hidden_states.append(hidden_states)
            # each_block_context_hidden_states.append(encoder_hidden_states)
        
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        encoder_hd_len=encoder_hidden_states.shape[1]

        for index_block, block in enumerate(self.single_transformer_blocks):
            
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            if hasattr(self,'control_single_trans_blocks'):
                interval_control = len(self.single_transformer_blocks) / len(self.control_single_trans_blocks)
                cn_block_idx=int(index_block / interval_control)
                moe_output['single_hd']=torch.cat([moe_output['control_encoder_hidden_states'],hidden_states[:, encoder_hd_len :, ...]],dim=1)
                zero_condition_hidden_states,condition_hidden_states,moe_output=self.control_forward(hidden_states,condition_hidden_states,None,
                                                    pooled_projections,timestep,conditioning_scale,condition_pooled_projections,
                                                    self.control_single_trans_blocks[cn_block_idx],self.controlnet_add_single_blocks[cn_block_idx],moe_output=moe_output,**kwargs)
                
                if self.single_block_control_method=="overall_add":
                    hidden_states=hidden_states+zero_condition_hidden_states
                else:
                    # in flux controlnet
                    zero_condition_hidden_states=zero_condition_hidden_states[:,encoder_hd_len:,...]
                    control_hidden_states=hidden_states[:,encoder_hd_len:,...]+zero_condition_hidden_states
                    hidden_states=torch.cat([hidden_states[:, :encoder_hd_len, ...],control_hidden_states],dim=1)
            
        hidden_states = hidden_states[:, encoder_hd_len :, ...]
            
        return dict(blocks_hidden_states=hidden_states,
                    block_ctx_hidden_states=encoder_hidden_states,
                    moe_loss=moe_output['moe_loss'],
                    exp_count=moe_output['exp_count']
                )
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        condition_hidden_states: torch.FloatTensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        condition_ids: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        skip_layers: Optional[List[int]] = None,
        **kwargs
    ):
        add_losses,add_outputs=dict(),dict()
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)  # takes care of adding positional embeddings too.

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        timestep = timestep.to(hidden_states.dtype) * 1000
        
        temb = self.time_text_embed(timestep, pooled_projections) if guidance is None else self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        forward_res=self.base_forward(hidden_states=hidden_states,
                                      condition_hidden_states=condition_hidden_states,
                                      encoder_hidden_states=encoder_hidden_states,
                                      pooled_projections=pooled_projections,
                                      condition_pooled_projections=condition_pooled_projections,
                                      timestep=timestep,
                                      conditioning_scale=conditioning_scale,
                                      temb=temb,
                                      joint_attention_kwargs=joint_attention_kwargs,
                                      image_rotary_emb=image_rotary_emb,
                                      guidance=guidance,
                                    #   control_encoder_hidden_states=control_encoder_hidden_states,
                                      img_ids=img_ids,prompt_ids=txt_ids,condition_ids=condition_ids
                                    )

        add_losses.update(dict(moe_loss=forward_res['moe_loss']*0.1))
        add_outputs.update(dict(expert_counts=forward_res['exp_count']))
        
        hidden_states = self.norm_out(forward_res['blocks_hidden_states'], temb)
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
            
        return hidden_states,add_losses,add_outputs
    

class MultiCondtionUniGenFlux(UniGenFlux):
    def preprocess_moe_forward(
        self,
        hidden_states,
        condition_hidden_states,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.LongTensor = None,
        timestep: torch.LongTensor = None,
        **kwargs
    ):  
        joint_attention_kwargs=dict()
        
        if self.use_pooled_prompt_embeds:
            control_pooled_projections=pooled_projections
        else:
            control_pooled_projections=torch.zeros_like(pooled_projections)
            
        control_temb=self.control_time_text_embed(timestep,control_pooled_projections) if kwargs['guidance'] is None else self.control_time_text_embed(timestep,kwargs['guidance'],control_pooled_projections)
        control_encoder_hidden_states=self.control_context_embedder(encoder_hidden_states)

        merge_hidden_states,merge_condition_temb=[],[]
        multi_cond_moe_losses,multi_cond_experts=[],[]
        for condition_id,condition_hidden_state,condition_pooled_embeds in zip(kwargs['condition_ids'],condition_hidden_states,condition_pooled_projections):
            # preprocess each condition infos
            if condition_pooled_embeds.ndim==1:
                condition_pooled_embeds=condition_pooled_embeds.unsqueeze(0)
            if condition_hidden_state.ndim==2:
                condition_hidden_state=condition_hidden_state.unsqueeze(0)

            condition_hidden_state=self.control_x_embedder(condition_hidden_state)
            condition_temb=self.control_condition_embed(timestep,condition_pooled_embeds) if kwargs['guidance'] is None else self.control_condition_embed(timestep,kwargs['guidance'],condition_pooled_embeds)
            
            if self.use_rope:
                joint_attention_kwargs=dict(img_ids=kwargs['img_ids'],prompt_ids=kwargs['prompt_ids'],condition_ids=condition_id,rope_embed=self.control_pos_embed_input)
            moe_output,moe_loss,exp_count=self.moe(hidden_states=hidden_states,condition_hidden_states=condition_hidden_state,encoder_hidden_states=control_encoder_hidden_states,temb=control_temb,condition_temb=condition_temb,condition_pooled_projections=condition_pooled_embeds,pooled_projections=pooled_projections,joint_attention_kwargs=joint_attention_kwargs)
        
            expert_hidden_states,expert_condition_hidden_states=moe_output
            
            merge_hidden_states.append(expert_hidden_states+expert_condition_hidden_states)
            merge_condition_temb.append(condition_temb)

        return dict(merged_hidden_states=sum(merge_hidden_states),
                    control_encoder_hidden_states=control_encoder_hidden_states,
                    control_temb=control_temb,
                    merged_condition_temb=sum(merge_condition_temb),
                    exp_count=exp_count,
                    moe_loss=moe_loss,
                    )
    
    def control_forward(
        self,                   
        hidden_states,
        condition_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        conditioning_scale: float = 1.0,
        condition_pooled_projections: torch.LongTensor = None,
        block: nn.Module = None,
        cn_block: nn.Module= None,
        **kwargs):
        joint_attention_kwargs=dict()
        
        moe_output=kwargs.pop('moe_output',None)
        if moe_output is None:
            moe_output=self.preprocess_moe_forward(hidden_states,condition_hidden_states,encoder_hidden_states,
                                            pooled_projections,condition_pooled_projections,timestep,**kwargs)
        
            hidden_states=moe_output['merged_hidden_states']
           
        encoder_hidden_states=moe_output['control_encoder_hidden_states']
        temb=moe_output['merged_condition_temb']
        
        if "single_hd" not in moe_output:
            if self.use_rope:
                joint_attention_kwargs=dict(hd_ids=kwargs['img_ids'],encoder_hd_ids=kwargs['prompt_ids'],rope_embed=self.control_pos_embed_input)
            encoder_hidden_states,hidden_states = block(hidden_states,encoder_hidden_states,temb,joint_attention_kwargs=joint_attention_kwargs)
        else:
            if self.use_rope:
                joint_attention_kwargs=dict(hd_ids=torch.cat((kwargs['prompt_ids'],kwargs['img_ids']),dim=0),rope_embed=self.control_pos_embed_input)
            hidden_states = block(hidden_states, temb,joint_attention_kwargs=joint_attention_kwargs)      

        return cn_block(hidden_states)*conditioning_scale,hidden_states,moe_output
    

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        condition_hidden_states: torch.FloatTensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        condition_ids: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        skip_layers: Optional[List[int]] = None,
        **kwargs
    ):
        add_losses,add_outputs=dict(),dict()
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)  # takes care of adding positional embeddings too.

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        timestep = timestep.to(hidden_states.dtype) * 1000
        
        temb = self.time_text_embed(timestep, pooled_projections) if guidance is None else self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)


        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        forward_res=self.base_forward(hidden_states=hidden_states,
                                      condition_hidden_states=condition_hidden_states,
                                      encoder_hidden_states=encoder_hidden_states,
                                      pooled_projections=pooled_projections,
                                      condition_pooled_projections=condition_pooled_projections,
                                      timestep=timestep,
                                      conditioning_scale=conditioning_scale,
                                      temb=temb,
                                      joint_attention_kwargs=joint_attention_kwargs,
                                      image_rotary_emb=image_rotary_emb,
                                      guidance=guidance,
                                    #   control_encoder_hidden_states=control_encoder_hidden_states,
                                      img_ids=img_ids,prompt_ids=txt_ids,condition_ids=condition_ids
                                    )

        add_losses.update(dict(moe_loss=forward_res['moe_loss']*0.1))
        add_outputs.update(dict(expert_counts=forward_res['exp_count']))
        
        hidden_states = self.norm_out(forward_res['blocks_hidden_states'], temb)
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
            
        return hidden_states,add_losses,add_outputs
 

class SANAUniGen(SanaTransformer2DModel):
    def init_condition_block(self,condition_nums=1,**kwargs): 
        self.condition_nums=condition_nums   
        self.init_control_block(kwargs.get('control_params',None))
    
    def init_control_block(self, control_params=None):
        assert control_params is not None, ValueError('Please provice control net model parameter')
        self.trainable_control_modules=dict()
        self.use_pooled_prompt_embeds=control_params.get('use_pooled_prompt_embeds',True)
        self.use_encoder_hidden_states=control_params.get('use_encoder_hidden_states',True)
        self.control_blocks_num=control_params.get('num_layers',self.config.num_layers)
        self.use_rope=control_params.get('use_rope',False)

        inner_dim=self.config.num_attention_heads* self.config.attention_head_dim
        self.inner_dim=inner_dim
        self.config.hidden_size=self.inner_dim
        
        # condition image patch embedding module
        from diffusers.models.embeddings import PatchEmbed
        self.control_pos_embed_input = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=self.config.interpolation_scale,
            pos_embed_type="sincos" if not self.use_rope and self.config.interpolation_scale is not None else None, # "sincos" if interpolation_scale is not None else None
        )
        self.trainable_control_modules.update(dict(control_pos_embed_input=self.control_pos_embed_input))
        
        # condition noise prediction time embedding module
        
        from diffusers.models.normalization import AdaLayerNormSingle

        # self.control_time_hd_pool_embed = CombinedTimestepTextProjEmbeddings(
        #     embedding_dim=self.inner_dim, pooled_projection_dim=control_params.pooled_projection_dim
        # )
        # self.control_time_condition_hd_pool_embed = CombinedTimestepTextProjEmbeddings(
        #     embedding_dim=self.inner_dim, pooled_projection_dim=control_params.pooled_projection_dim
        # )
        
        self.control_condition_embed = AdaLayerNormSingle(self.inner_dim)
        self.trainable_control_modules.update(dict(control_condition_embed=self.control_condition_embed))
        
        # condition context embedding module
        from diffusers.models.embeddings import PixArtAlphaTextProjection
        # self.control_context_embedder = PixArtAlphaTextProjection(in_features=self.config.caption_channels, hidden_size=self.inner_dim)
        self.control_context_embedder = nn.Linear(self.inner_dim, self.inner_dim)
        self.trainable_control_modules.update(dict(control_context_embedder=self.control_context_embedder))
        
        # from diffusers.models.normalization import RMSNorm
        # self.control_context_norm=RMSNorm(self.inner_dim, eps=1e-5, elementwise_affine=True)
        # self.trainable_control_modules.update(dict(control_context_norm=self.control_context_norm))
        
        if not self.use_rope:
            from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
            transblock=SanaTransformerBlock
        else:
            from src.UniGenUtils import SanaRoPETransformerBlock
            transblock=SanaRoPETransformerBlock

        self.control_transformer_blocks=nn.ModuleList([
            transblock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    num_cross_attention_heads=self.config.num_cross_attention_heads,
                    cross_attention_head_dim=self.config.cross_attention_head_dim,
                    cross_attention_dim=self.config.cross_attention_dim,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    mlp_ratio=self.config.mlp_ratio,
                    qk_norm=self.config.qk_norm,
                ) for _ in range(self.control_blocks_num)
        ])
        self.trainable_control_modules.update(dict(control_transformer_blocks=self.control_transformer_blocks))

        # control net attention block out add to mmdit
        self.controlnet_add_blocks = nn.ModuleList([])
        for _ in range(self.control_blocks_num):
            self.controlnet_add_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))
            
        self.trainable_control_modules.update(dict(controlnet_add_blocks=self.controlnet_add_blocks))
        
        if control_params.get('use_transformer_params',False):
            self.init_control_param()
        debug_print(f'Init controlnet block success......')
        
        if self.use_rope:
            from diffusers.models.embeddings import FluxPosEmbed
            self.rope_embed=FluxPosEmbed(theta=10000,axes_dim=(8, 28, 28))
        
        self.init_moe_block(self.inner_dim,self.config.num_attention_heads, self.config.attention_head_dim,control_params)

        self.cn_method=control_params.get("cn2base_method","add")
    
    def init_trainable_param(self):
        for name,module in self.trainable_control_modules.items():
            module.requires_grad_(True)
    
    def init_control_param(self):
        self.control_pos_embed_input.load_state_dict(self.patch_embed.state_dict(),strict=False)
        
        self.control_condition_embed.load_state_dict(self.time_embed.state_dict())
        
        self.control_transformer_blocks.load_state_dict(self.transformer_blocks.state_dict(),strict=False)
        debug_print('Initialize controlnet based on Transformer parameters....')
    
    def init_moe_block(self,inner_dim,num_attention_heads, attention_head_dim,control_params):
        self.expert_nums=control_params.get('expert_num',None) if control_params.get('expert_num',None) is not None else (self.condition_nums+1)*control_params.get('expert_num_each_condition',3)
        self.top_k=control_params.get('top_num',1)
        
        if not self.use_rope:
            from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
            transblock=SanaTransformerBlock
        else:
            from src.UniGenUtils import SanaRoPETransformerBlock
            transblock=SanaRoPETransformerBlock
            
        # self.preprocess_block=nn.ModuleList([
        #     transblock(
        #                 self.inner_dim,
        #                 self.config.num_attention_heads,
        #                 self.config.attention_head_dim,
        #                 dropout=self.config.dropout,
        #                 num_cross_attention_heads=self.config.num_cross_attention_heads,
        #                 cross_attention_head_dim=self.config.cross_attention_head_dim,
        #                 cross_attention_dim=self.config.cross_attention_dim,
        #                 attention_bias=self.config.attention_bias,
        #                 norm_elementwise_affine=self.config.norm_elementwise_affine,
        #                 norm_eps=self.config.norm_eps,
        #                 mlp_ratio=self.config.mlp_ratio,
        #                 qk_norm=self.config.qk_norm,
        #             ),
        #     transblock(
        #                 self.inner_dim,
        #                 self.config.num_attention_heads,
        #                 self.config.attention_head_dim,
        #                 dropout=self.config.dropout,
        #                 num_cross_attention_heads=self.config.num_cross_attention_heads,
        #                 cross_attention_head_dim=self.config.cross_attention_head_dim,
        #                 cross_attention_dim=self.config.cross_attention_dim,
        #                 attention_bias=self.config.attention_bias,
        #                 norm_elementwise_affine=self.config.norm_elementwise_affine,
        #                 norm_eps=self.config.norm_eps,
        #                 mlp_ratio=self.config.mlp_ratio,
        #                 qk_norm=self.config.qk_norm,
        #             )
        # ])
        # self.trainable_control_modules.update(dict(preprocess_block=self.preprocess_block))
        
        self.use_modulate=control_params.get('use_modulate',False)
        if self.use_modulate or self.use_rope:
            moe_init_denoise=nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(inner_dim,inner_dim),
                    nn.Linear(control_params.pooled_projection_dim,inner_dim)
                ]),
                nn.ModuleList([
                    nn.Linear(inner_dim,inner_dim),
                    nn.Linear(control_params.pooled_projection_dim,inner_dim)
                ])                
            ])
        else:
            moe_init_denoise=nn.ModuleList([
                transblock(
                        self.inner_dim,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        dropout=self.config.dropout,
                        num_cross_attention_heads=self.config.num_cross_attention_heads,
                        cross_attention_head_dim=self.config.cross_attention_head_dim,
                        cross_attention_dim=self.config.cross_attention_dim,
                        attention_bias=self.config.attention_bias,
                        norm_elementwise_affine=self.config.norm_elementwise_affine,
                        norm_eps=self.config.norm_eps,
                        mlp_ratio=self.config.mlp_ratio,
                        qk_norm=self.config.qk_norm,
                    ), # for hidden states 
                transblock(
                        self.inner_dim,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        dropout=self.config.dropout,
                        num_cross_attention_heads=self.config.num_cross_attention_heads,
                        cross_attention_head_dim=self.config.cross_attention_head_dim,
                        cross_attention_dim=self.config.cross_attention_dim,
                        attention_bias=self.config.attention_bias,
                        norm_elementwise_affine=self.config.norm_elementwise_affine,
                        norm_eps=self.config.norm_eps,
                        mlp_ratio=self.config.mlp_ratio,
                        qk_norm=self.config.qk_norm,
                    ) # for condition - hidden_states
            ])
        
        self.moe=MoE(inner_dim,moe_init_denoise,self.expert_nums,k=self.top_k)
        self.moe.moe_layer.experts.forward=self.expert_forward
        self.moe.forward=self.moe_forward
        self.num_local_experts=self.moe.num_local_experts
        self.trainable_control_modules.update(dict(moe=self.moe))
        
        self.use_shared_expert=control_params.get('use_shared_expert',False)
        if self.use_shared_expert:
            # V1
            """
            self.shared_expert = jointtransblock(
                                    dim=inner_dim,
                                    num_attention_heads=num_attention_heads,
                                    attention_head_dim=attention_head_dim,
                                    context_pre_only=False,
                                    qk_norm=control_params.get('qk_norm',self.config.qk_norm),
                                    use_dual_attention=False,
                                )
            # coefficient is used for weighted sum of the output of expert and mlp
            # self.coefficient = nn.Linear(inner_dim, 2)
            """
            # V2
            self.shared_expert = nn.ModuleList([
                transblock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    num_cross_attention_heads=self.config.num_cross_attention_heads,
                    cross_attention_head_dim=self.config.cross_attention_head_dim,
                    cross_attention_dim=self.config.cross_attention_dim,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    mlp_ratio=self.config.mlp_ratio,
                    qk_norm=self.config.qk_norm,
                ) # use encoder hidden states to constrain hidden states and condition hidden states
            ])
            self.trainable_control_modules.update(dict(shared_expert=self.shared_expert))
        
        self.use_consis_module=control_params.get('use_consis_module',False)
        if self.use_consis_module:
            """
            # V1
            self.consis_module=nn.ModuleList([
                singletransblock(
                    inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim
                ),
                singletransblock(
                    inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim
                )
            ])
            """
            # V2
            self.consis_module=nn.ModuleList([
                transblock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    num_cross_attention_heads=self.config.num_cross_attention_heads,
                    cross_attention_head_dim=self.config.cross_attention_head_dim,
                    cross_attention_dim=self.config.cross_attention_dim,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    mlp_ratio=self.config.mlp_ratio,
                    qk_norm=self.config.qk_norm,
                ),
                transblock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    num_cross_attention_heads=self.config.num_cross_attention_heads,
                    cross_attention_head_dim=self.config.cross_attention_head_dim,
                    cross_attention_dim=self.config.cross_attention_dim,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    mlp_ratio=self.config.mlp_ratio,
                    qk_norm=self.config.qk_norm,
                )
            ])
            self.trainable_control_modules.update(dict(consis_module=self.consis_module))
        
    def expert_forward(
        self,
        hidden_states,
        condition_hidden_states: torch.Tensor = None,
        temb: torch.Tensor = None,
        condition_temb: torch.Tensor = None,
        **kwargs
    ):
        expert_hidden_out: List[torch.Tensor] = []
        expert_condition_hidden_out: List[torch.Tensor] = []
        
        hidden_chunks = hidden_states.chunk(self.num_local_experts, dim=1)
        condition_chunks=condition_hidden_states.chunk(self.num_local_experts,dim=1)
        temb_chunks=temb.chunk(self.num_local_experts,dim=1)
        condition_temb_chunks=condition_temb.chunk(self.num_local_experts,dim=1)
        
        condition_pooled_embeds=kwargs.get('condition_pooled_projections',None)
        pooled_embeds=kwargs.get('pooled_projections',None)
        
        if self.use_modulate: assert condition_pooled_embeds is not None and pooled_embeds is not None, ValueError('Check MoE Forward input, condition pooled projection or pooled projection is None!')
        
        condition_pooled_chunks=condition_pooled_embeds.chunk(self.num_local_experts,dim=1)
        pooled_chunks=pooled_embeds.chunk(self.num_local_experts,dim=1)
        
        for expert_idx,(hidden_chunk,condition_chunk,temb_chunk,condition_temb_chunk, expert) in enumerate(zip(hidden_chunks,condition_chunks,temb_chunks,condition_temb_chunks,self.moe.moe_layer.experts.deepspeed_experts)):
            hidden_chunk,condition_chunk,temb_chunk,condition_temb_chunk=hidden_chunk.squeeze(1),condition_chunk.squeeze(1),temb_chunk.squeeze(1),condition_temb_chunk.squeeze(1)
            
            if self.use_modulate or self.use_rope:
                condition_pooled_chunk=condition_pooled_chunks[expert_idx].squeeze(1)
                pooled_chunk=pooled_chunks[expert_idx].squeeze(1)
                
                condition_modulate,hidden_modulate=expert[0],expert[1]
                condition_chunk=modulated_flatten(condition_chunk,condition_modulate[0].weight,condition_modulate[1](condition_pooled_chunk))+condition_modulate[0].bias.unsqueeze(0)
                
                hidden_chunk=modulated_flatten(hidden_chunk+condition_chunk,hidden_modulate[0].weight,hidden_modulate[1](pooled_chunk))+hidden_modulate[0].bias.unsqueeze(0)
            else:
                hidden_chunk=expert[0](hidden_chunk,temb_chunk)
                condition_chunk=expert[1](condition_chunk,condition_temb_chunk)
            
            expert_hidden_out.append(hidden_chunk)
            expert_condition_hidden_out.append(condition_chunk)

        return (torch.stack(expert_hidden_out, dim=1),torch.stack(expert_condition_hidden_out, dim=1))
    
    def moe_forward(self,
                    hidden_states: torch.Tensor,
                    condition_hidden_states: torch.Tensor,
                    used_token: Optional[torch.Tensor] = None,
                    **kwargs):
        joint_attention_kwargs=kwargs.pop('joint_attention_kwargs',dict())
        if self.use_rope:
            assert len(joint_attention_kwargs.keys())!=0, ValueError('Please give joint_attention_kwargs info when use rope!')
        else:
            new_joint_attention_kwargs=dict()
        expert_output=self.moe.moe_layer(choice_expert_input=hidden_states+condition_hidden_states,hidden_states=hidden_states,condition_hidden_states=condition_hidden_states,used_token=used_token,**kwargs)
        
        expert_hidden_states,expert_condition_states=expert_output
        if self.use_consis_module:
            """
            # V1
            consis_expert_hidden=self.consis_module[0](expert_hidden_states,temb=kwargs['temb'])
            consis_expert_condition_hidden=self.consis_module[1](expert_condition_states,temb=kwargs['condition_temb'])
            
            expert_hidden_states=expert_hidden_states+consis_expert_hidden
            expert_condition_states=expert_condition_states+consis_expert_condition_hidden
            """
            #V2
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=joint_attention_kwargs['condition_ids'],encoder_hd_ids=joint_attention_kwargs['condition_ids'],rope_embed=self.rope_embed)
            _,consis_expert_condition_hidden=self.consis_module[0](expert_condition_states,condition_hidden_states,temb=kwargs['condition_temb'],joint_attention_kwargs=new_joint_attention_kwargs)
            
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=torch.cat([joint_attention_kwargs['img_ids'],joint_attention_kwargs['condition_ids']],dim=0),encoder_hd_ids=joint_attention_kwargs['img_ids'],rope_embed=self.rope_embed)
            _,consis_expert_hidden_condition_states=self.consis_module[0](torch.cat([expert_hidden_states,consis_expert_condition_hidden],dim=1),hidden_states,temb=kwargs['temb'],joint_attention_kwargs=new_joint_attention_kwargs)
            
            consis_expert_hidden,consis_expert_condition_hidden=consis_expert_hidden_condition_states[:,:expert_hidden_states.shape[1],:],consis_expert_hidden_condition_states[:,expert_hidden_states.shape[1]:,:]
            
            expert_hidden_states=expert_hidden_states+consis_expert_hidden
            expert_condition_states=expert_condition_states+consis_expert_condition_hidden
            
        if self.use_shared_expert:
            # V1
            """
            hidden_encoder_states=torch.cat([hidden_states,kwargs['encoder_hidden_states']],dim=1)
            condition_states,hidden_encoder_states = self.shared_expert(hidden_encoder_states,kwargs['condition_hidden_states'],temb=kwargs['temb']+kwargs['condition_temb'])
            hidden_states,encoder_hidden_states=hidden_encoder_states[:,:expert_hidden_states.shape[1],:],hidden_encoder_states[:,expert_hidden_states.shape[1]:,:]
            """
            # V2
            # if self.use_rope:
            #     new_joint_attention_kwargs=dict(hd_ids=joint_attention_kwargs['img_ids'],encoder_hd_ids=joint_attention_kwargs['condition_ids'],rope_embed=self.pos_embed)
            # condition_states,hidden_states=self.shared_expert[0](hidden_states,
            #                                                      attention_mask=None,
            #                                                      encoder_hidden_states=condition_hidden_states,
            #                                                      encoder_attention_mask=None,
            #                                                      timestep=kwargs['temb'],
            #                                                      height=kwargs['height'],
            #                                                      width=kwargs['width'],
            #                                                      joint_attention_kwargs=new_joint_attention_kwargs)
            
            if self.use_rope:
                new_joint_attention_kwargs=dict(hd_ids=torch.cat([joint_attention_kwargs['img_ids'],joint_attention_kwargs['condition_ids']],dim=0),encoder_hd_ids=joint_attention_kwargs['prompt_ids'],rope_embed=self.rope_embed)
            
            hidden_condition_states=torch.cat([hidden_states,condition_hidden_states],dim=1)
            
            hidden_condition_states=self.shared_expert[0](hidden_condition_states,
                                                            attention_mask=None,
                                                            encoder_hidden_states=kwargs['encoder_hidden_states'],
                                                            encoder_attention_mask=kwargs['encoder_attention_mask'],
                                                            timestep=kwargs['temb'],
                                                            height=kwargs['height'],
                                                            width=kwargs['width'],
                                                            joint_attention_kwargs=new_joint_attention_kwargs)
            
            hidden_states,condition_states=hidden_condition_states[:,:hidden_states.shape[1],:],hidden_condition_states[:,hidden_states.shape[1]:,:]
            
            expert_output=(hidden_states+expert_hidden_states,condition_states+expert_condition_states)
            
        return expert_output, self.moe.moe_layer.l_aux, self.moe.moe_layer.exp_counts
    
    def preprocess_moe_forward(
        self,
        hidden_states,
        condition_hidden_states,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.LongTensor = None,
        timestep: torch.LongTensor = None,
        **kwargs
    ):  
        
        condition_hidden_states=self.control_pos_embed_input(condition_hidden_states)
        
        control_proj_timestep, _ = self.control_condition_embed(
            timestep, batch_size=condition_hidden_states.shape[0], hidden_dtype=hidden_states.dtype
        )

        control_encoder_hidden_states=self.control_context_embedder(encoder_hidden_states)
        
        moe_output,moe_loss,exp_count=self.moe(hidden_states=hidden_states,
                                               condition_hidden_states=condition_hidden_states,
                                               encoder_hidden_states=control_encoder_hidden_states,
                                               temb=control_proj_timestep,
                                               condition_pooled_projections=condition_pooled_projections,
                                               pooled_projections=pooled_projections,
                                               joint_attention_kwargs=kwargs)
        
        expert_hidden_states,expert_condition_hidden_states=moe_output
        
        return dict(
                    condition_hidden_states=condition_hidden_states,
                    expert_hidden_states=expert_hidden_states,
                    expert_condition_hidden_states=expert_condition_hidden_states,
                    control_encoder_hidden_states=control_encoder_hidden_states,
                    condition_temb=control_proj_timestep,
                    exp_count=exp_count,
                    moe_loss=moe_loss,
                    )
    
    def control_forward(
        self,                   
        hidden_states,
        attention_mask,
        condition_hidden_states: torch.FloatTensor = None,
        condition_attention_mask: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        conditioning_scale: float = 1.0,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.LongTensor = None,
        block: nn.Module = None,
        cn_block: nn.Module= None,
        **kwargs):
        joint_attention_kwargs=dict()

        moe_output=kwargs.pop('moe_output',None)
        if moe_output is None:
            moe_output=self.preprocess_moe_forward(
                hidden_states,
                condition_hidden_states,
                encoder_hidden_states,
                pooled_projections,
                condition_pooled_projections,
                timestep, 
                attention_mask=attention_mask,
                condition_attention_mask=condition_attention_mask,
                encoder_attention_mask=encoder_attention_mask,**kwargs)
        
            hidden_states=moe_output['expert_hidden_states']+moe_output['expert_condition_hidden_states']
            
        encoder_hidden_states=moe_output['control_encoder_hidden_states']
        temb=moe_output['condition_temb']
        
        if self.use_rope:
            joint_attention_kwargs=dict(hd_ids=kwargs['img_ids'],encoder_hd_ids=kwargs['prompt_ids'],rope_embed=self.rope_embed)
        
        hidden_states = block(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            temb,
            post_patch_height=kwargs['height'],
            post_patch_width=kwargs['width'],
            joint_attention_kwargs=joint_attention_kwargs)
    
        return cn_block(hidden_states)*conditioning_scale,hidden_states,moe_output
    
    def base_forward(
        self,
        hidden_states,
        attention_mask,
        condition_hidden_states,
        condition_attention_mask,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        pooled_projections: torch.Tensor = None,
        condition_pooled_projections: torch.Tensor = None,
        proj_timestep: torch.Tensor = None,
        post_patch_height: int = None,
        post_patch_width: int = None,
        conditioning_scale: float = 1.0,
        **kwargs):

        moe_output=None

        for index_block, block in enumerate(self.transformer_blocks):
            
            interval_control = len(self.transformer_blocks) / len(self.control_transformer_blocks)
            cn_block_idx=int(index_block / interval_control)
            
            hidden_states = block(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                proj_timestep,
                post_patch_height,
                post_patch_width,
            )
            
            zero_condition_hidden_states,condition_hidden_states,moe_output=self.control_forward(
                hidden_states,
                attention_mask,
                condition_hidden_states,
                condition_attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                kwargs['timestep'],
                conditioning_scale,
                pooled_projections,
                condition_pooled_projections,
                self.control_transformer_blocks[cn_block_idx],
                self.controlnet_add_blocks[cn_block_idx],
                moe_output=moe_output,
                height=post_patch_height,width=post_patch_width,**kwargs)
            
            hidden_states=hidden_states+zero_condition_hidden_states
            
        return dict(blocks_hidden_states=hidden_states,
                    block_ctx_hidden_states=encoder_hidden_states,
                    moe_loss=moe_output['moe_loss'],
                    exp_count=moe_output['exp_count']
                )
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        condition_hidden_states: torch.FloatTensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        condition_attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        add_losses,add_outputs=dict(),dict()
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        
        if condition_attention_mask is not None and condition_attention_mask.ndim == 2:
            condition_attention_mask = (1 - condition_attention_mask.to(hidden_states.dtype)) * -10000.0
            condition_attention_mask = condition_attention_mask.unsqueeze(1)
        
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)

        if self.use_rope:
            from src.UniGenUtils import prepare_latent_image_ids
            img_ids=prepare_latent_image_ids(post_patch_height,post_patch_width,hidden_states.device,hidden_states.dtype)
            prompt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
            condition_ids=prepare_latent_image_ids(post_patch_height,post_patch_width,device=condition_hidden_states.device,dtype=condition_hidden_states.dtype)
        else:
            img_ids,prompt_ids,condition_ids=None,None,None

        proj_timestep, embedded_timestep = self.time_embed(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        forward_res=self.base_forward(hidden_states=hidden_states,
                                      attention_mask=attention_mask,
                                      condition_hidden_states=condition_hidden_states,
                                      condition_attention_mask=condition_attention_mask,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=encoder_attention_mask,
                                      pooled_projections=pooled_projections,
                                      condition_pooled_projections=condition_pooled_projections,
                                      proj_timestep=proj_timestep,
                                      post_path_height=post_patch_height,
                                      post_path_width=post_patch_width,
                                      conditioning_scale=conditioning_scale,
                                      img_ids=img_ids,prompt_ids=prompt_ids,condition_ids=condition_ids,
                                      timestep=timestep
                                    )

        add_losses.update(dict(moe_loss=forward_res['moe_loss']*0.1))
        add_outputs.update(dict(expert_counts=forward_res['exp_count']))
        
        # 3. Normalization
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)

        # 4. Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        return output,add_losses,add_outputs



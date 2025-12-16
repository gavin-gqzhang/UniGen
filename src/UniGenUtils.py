import torch
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, List
from torch import FloatTensor, Tensor
import torch.nn.functional as F
from torch import nn
import os
from deepspeed.utils import groups
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size
from deepspeed.moe.experts import Experts
from deepspeed.moe import sharded_moe,mappings
from diffusers.models.attention_processor import JointAttnProcessor2_0,Attention
from diffusers.models.transformers.transformer_sd3 import SD3SingleTransformerBlock as SD3TransBlock
from diffusers.models import attention as diffusers_attention
from diffusers.models.embeddings import FluxPosEmbed,apply_rotary_emb
import functools

class MoE(nn.Module):
    def __init__(self, hidden_size, expert, num_experts = 1, ep_size = 1, k = 1, capacity_factor = 1, eval_capacity_factor = 1, min_capacity = 4, use_residual = False, noisy_gate_policy = None, drop_tokens = True, use_rts = True, use_tutel = False, enable_expert_tensor_parallelism = False, top2_2nd_expert_sampling = True,share_expert=None):
        super().__init__()
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        debug_print(
            f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size} | Gate top k: {k}')

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.moe_layer = MOELayer(sharded_moe.TopKGate(hidden_size, num_experts, k, capacity_factor, eval_capacity_factor,
                                        min_capacity, noisy_gate_policy, drop_tokens, use_rts, None,
                                        top2_2nd_expert_sampling),
                                experts,
                                self.expert_group_name,
                                self.ep_size,
                                self.num_local_experts,
                                use_tutel=use_tutel)
        
    def set_deepspeed_parallelism(self, use_data_before_expert_parallel_: bool = False) -> None:
        self._create_process_groups(use_data_before_expert_parallel_=use_data_before_expert_parallel_)

    def _create_process_groups(self, use_data_before_expert_parallel_: bool = False) -> None:
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(
                    self.ep_size, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(
                    self.ep_size, mpu=groups.mpu, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.moe_layer._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self,
                hidden_states: torch.Tensor,
                used_token: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise ValueError('Please initialize the MoE module in the main class and redirect and rewrite the forward of the MoE module!')

class MOELayer(sharded_moe.MOELayer):
    def __init__(self, gate, experts, ep_group_name, ep_size, num_local_experts, use_tutel = False):
        super().__init__(gate, experts, ep_group_name, ep_size, num_local_experts, use_tutel)
        
        # self.gate.forward=self.gate_forward
        
    def forward(self, **kwargs: Any) -> Tensor:
        
        if self.wall_clock_breakdown:
            self.timers(sharded_moe.MOE_TIMER).start()

        # Implement Algorithm 2 from GShard paper.
        choice_expert_input=kwargs.pop('choice_expert_input',None)
        if choice_expert_input is None:
            assert ValueError('Please provide the choice_expert_input parameter in MOE Layer Forward!')
        
        d_model = choice_expert_input.shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_input = choice_expert_input.reshape(-1, d_model)

        if self.use_tutel:
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, kwargs.get('used_token',None), True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
        else:
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, kwargs.get('used_token',None)) # num_tokens, expert_num,capacity
        
        tensor_model_world_size = bwc_tensor_model_parallel_world_size(groups.mpu)
        
        dispatched_kwargs=dict()
        for name,value in kwargs.items():
            if isinstance(value, torch.Tensor):
                v_c=value.shape[-1]
                if len(value.shape)==2:
                    value=value.unsqueeze(1).expand(-1,choice_expert_input.shape[1],-1).reshape(-1,v_c)
                    dispatched_value = self.input_check_expert(v_c,value,tensor_model_world_size,dispatch_mask if not self.use_tutel else None) # (expert_num,capacity_num,hidden_states)
                elif len(value.shape)==3:
                    if value.shape[1]!=choice_expert_input.shape[1]: 
                        dispatched_kwargs[name]=value
                        continue
                    dispatched_value=self.input_check_expert(v_c,value.reshape(-1,v_c),tensor_model_world_size,dispatch_mask if not self.use_tutel else None)
                else:
                    raise ValueError(f'MoE Layer forward get a error input, name: {name}, value shape: {value.shape}')

                dispatched_kwargs[name]=dispatched_value
            else:
                dispatched_kwargs[name]=value
        
        
        expert_output = self.experts(**dispatched_kwargs)

        if not isinstance(expert_output,(list,tuple)):
            reshape_expert_outputs=self.postprocess_output(choice_expert_input,expert_output,d_model,tensor_model_world_size,combine_weights,E if self.use_tutel else None,C if self.use_tutel else None,M if self.use_tutel else None)
            reshape_expert_outputs=(reshape_expert_outputs,)
        else:
            reshape_expert_outputs=[]
            for output in expert_output:
                reshape_expert_output=self.postprocess_output(choice_expert_input,output,d_model,tensor_model_world_size,combine_weights,E if self.use_tutel else None,C if self.use_tutel else None,M if self.use_tutel else None)
                reshape_expert_outputs.append(reshape_expert_output)

        return tuple(reshape_expert_outputs)

    def input_check_expert(self,d_model,reshaped_input,tensor_model_world_size,dispatch_mask=None):
        if self.use_tutel:
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            dispatched_input = sharded_moe.einsum("sec,sm->ecm", dispatch_mask.type_as(reshaped_input), reshaped_input) # (expert_num,capacity_num,hidden_states)
        
        if self.wall_clock_breakdown:
            self.timers(sharded_moe.FIRST_ALLTOALL_TIMER).start()
        
        if tensor_model_world_size > 1:
            dispatched_input = mappings.drop_tokens(dispatched_input, dim=1)
            
        if self.ep_group is not None:
            dispatched_input = sharded_moe._AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers(sharded_moe.FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(sharded_moe.FIRST_ALLTOALL_TIMER).elapsed(reset=False)

        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            dispatched_input = mappings.gather_tokens(dispatched_input, dim=1)
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)
        
        return dispatched_input
        
    def postprocess_output(self,input,expert_output,d_model,tensor_model_world_size,combine_weights,E,C,M):
        # Re-shape before drop_tokens: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)
        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            expert_output = mappings.drop_tokens(expert_output, dim=1)

        if self.wall_clock_breakdown:
            self.timers(sharded_moe.SECOND_ALLTOALL_TIMER).start()

        if self.ep_group is not None:
            expert_output = sharded_moe._AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers(sharded_moe.SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(sharded_moe.SECOND_ALLTOALL_TIMER).elapsed(reset=False)

        if tensor_model_world_size > 1:
            expert_output = mappings.gather_tokens(expert_output, dim=1)

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = sharded_moe.einsum("sec,ecm->sm", combine_weights.type_as(input), expert_output)

        a = combined_output.reshape(input.shape)

        if self.wall_clock_breakdown:
            self.timers(sharded_moe.MOE_TIMER).stop()
            self.time_moe = self.timers(sharded_moe.MOE_TIMER).elapsed(reset=False)

        return a
    
    
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def debug_print(info):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank==0:
        print(info)
        
def modulated_flatten(
    x,                 
    w,                 
    s,                 
):
    
    bs, nl, in_c = x.shape
    out_channels = w.shape[0]

    # Modulate weights.
    if len(s.shape)==2:
        w = w.unsqueeze(0) # 1, o_c, i_c
        w = (w * s.unsqueeze(1)) # (1, o_c, i_c) * (b, 1, i_c)
        
        x=x.transpose(-1,-2).reshape(1, -1, nl)  # 1,b*i_c,nl
        w=w.reshape(-1, in_c, 1) # b*o_c, i_c, 1
        
        x=torch.nn.functional.conv1d(x,w,groups=bs)
        x=x.reshape(bs, out_channels, nl).transpose(-1,-2)
    else:
        w=w.unsqueeze(0).unsqueeze(1)  
        w=(w*s.unsqueeze(2))   # (1,1,o_c,i_c) * (b,n,1,i_c)
        x=torch.einsum('bnoi, bni->bno',w,x)

    return x

from torch.utils.data import BatchSampler,Sampler
import math
class MultiTaskMixedBatchSampler(BatchSampler):
    def __init__(self, datasets, batch_size, num_replicas=None, rank=None, shuffle=True, seed=42, drop_last=False, sampler=None):
        """
        Args:
            datasets (List[Dataset]): List of individual datasets.
            batch_size (int): Total batch size.
        """
        self.datasets = datasets.datasets
        self.batch_size = batch_size
        
        self.drop_last=drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        
        self.num_datasets = len(self.datasets)
        self.dataset_lengths = [len(ds) for ds in self.datasets]
        self.max_length=max(self.dataset_lengths)
        self.total_samples = self.max_length * self.num_datasets
        
        self.dataset_ranges = []
        start = 0
        for length in self.dataset_lengths:
            self.dataset_ranges.append(range(start, start + length))
            start += length
        
        # each rank sample numbers and batch size
        self.samples_per_replica = math.ceil(self.total_samples / self.num_replicas)
        self.local_batch_size = self.batch_size // self.num_replicas
        
        self.task_indices_cache = self._prepare_task_indices()

    def _prepare_task_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        task_indices = []
        for dataset_range, dataset_len in zip(self.dataset_ranges, self.dataset_lengths):
            indices = list(dataset_range)
            repeat_factor = math.ceil(self.max_length / dataset_len)
            indices = (indices * repeat_factor)[:self.max_length]
            if self.shuffle:
                indices = [indices[j] for j in torch.randperm(len(indices), generator=g)]
            task_indices.append(indices)
        return task_indices
    
    def __len__(self):
        # return self.num_samples_per_replica
        if self.drop_last:
            return self.samples_per_replica // self.local_batch_size
        return math.ceil(self.samples_per_replica / self.local_batch_size)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed+self.rank)

        task_cursors = [0] * len(self.task_indices_cache)
        local_batches = []
        collected = 0

        task_indices_per_rank = []
        for task_id in range(self.num_datasets):
            indices = self.task_indices_cache[task_id]

            indices_for_rank = indices[self.rank::self.num_replicas]
            task_indices_per_rank.append(indices_for_rank)
    
        while collected < self.samples_per_replica:
            batch = []
            
            task_ids = list(range(len(self.task_indices_cache)))
            if self.shuffle:
                task_ids = [task_ids[i] for i in torch.randperm(len(task_ids), generator=g)]

            per_task = self.local_batch_size // self.num_datasets
            extra = self.local_batch_size % self.num_datasets

            for i, task_id in enumerate(task_ids):
                n = per_task + (1 if i < extra else 0)
                for _ in range(n):
                    idx_list = task_indices_per_rank[task_id]
                    cursor = task_cursors[task_id]
                    if cursor >= len(idx_list):
                        new_indices = idx_list.copy()
                        if self.shuffle:
                            new_indices = [new_indices[j] for j in torch.randperm(len(new_indices), generator=g)]
                        task_indices_per_rank[task_id] = new_indices
                        task_cursors[task_id] = 0
                        cursor = 0
                        
                    batch.append(task_indices_per_rank[task_id][cursor])
                    task_cursors[task_id] += 1
            
            if len(batch) == 0:
                break
                
            if len(batch)<self.local_batch_size and self.drop_last:
                break

            if self.shuffle:
                batch = [batch[i] for i in torch.randperm(len(batch), generator=g)]
            local_batches.append(batch)
            collected += len(batch)

        return iter(local_batches)

def sd35adanormX_forward(module, hidden_states, emb):
    emb = module.linear(module.silu(emb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(
        9, dim=-1
    )
    norm_hidden_states = module.norm(hidden_states)
    if len(hidden_states.shape)==len(emb.shape):
        hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2) + shift_msa2
    else:
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
    return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2

def adanorm_forward(module, hidden_states, timestep=None, class_labels=None, hidden_dtype=None, emb=None):
    if module.emb is not None:
        emb = module.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
    emb = module.linear(module.silu(emb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
    if len(emb.shape)==len(hidden_states.shape):
        hidden_states = module.norm(hidden_states) * (1 + scale_msa) + shift_msa
    else:
        hidden_states = module.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp

def adanormContinuous_forward(module, hidden_states: torch.Tensor, emb: torch.Tensor):
    # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
    emb = module.linear(module.silu(emb).to(hidden_states.dtype))
    scale, shift = torch.chunk(emb, 2, dim=1)
    if len(hidden_states.shape)==len(emb.shape):
        hidden_states = module.norm(hidden_states) * (1 + scale) + shift
    else:
        hidden_states = module.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
    return hidden_states

class SD3SingleTransformerBlock(SD3TransBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__(dim, num_attention_heads, attention_head_dim)

        if isinstance(self.norm1,diffusers_attention.SD35AdaLayerNormZeroX):
            self.norm1.forward=functools.partial(sd35adanormX_forward,module=self.norm1)
        elif isinstance(self.norm1,diffusers_attention.AdaLayerNormZero):
            self.norm1.forward=functools.partial(adanorm_forward,module=self.norm1)
        else:
            raise ValueError(f'Please check SD3SingleTransformerBlock norm1 type: {type(self.norm1)}')
    
    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, joint_attention_kwargs: Optional[Dict[str, Any]] = None):
        joint_attention_kwargs = joint_attention_kwargs or {}
        expand_gate_dim=len(hidden_states.shape)!=len(temb.shape)
        
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states=hidden_states, emb=temb)
        # Attention.
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            **joint_attention_kwargs
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output if expand_gate_dim else gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        
        if expand_gate_dim:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        else:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output if expand_gate_dim else gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        return hidden_states

class SingleRoPETransformerBlock(SD3SingleTransformerBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__(dim, num_attention_heads, attention_head_dim)
        
        self.attn.processor=JointAttnRopeProcessor()

class JointTransformerBlock(diffusers_attention.JointTransformerBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only = False, qk_norm = None, use_dual_attention = False):
        super().__init__(dim, num_attention_heads, attention_head_dim, context_pre_only, qk_norm, use_dual_attention)

        if isinstance(self.norm1,diffusers_attention.SD35AdaLayerNormZeroX):
            self.norm1.forward=functools.partial(sd35adanormX_forward,module=self.norm1)
        elif isinstance(self.norm1,diffusers_attention.AdaLayerNormZero):
            self.norm1.forward=functools.partial(adanorm_forward,module=self.norm1)
        else:
            raise ValueError(f'Please check JointRoPETransformerBlock or JointTransformerBlock norm1 type: {type(self.norm1)}')
        
        if isinstance(self.norm1_context,diffusers_attention.AdaLayerNormZero):
            self.norm1_context.forward=functools.partial(adanorm_forward,module=self.norm1_context)
        elif isinstance(self.norm1_context,diffusers_attention.AdaLayerNormContinuous):
            self.norm1_context.forward=functools.partial(adanormContinuous_forward,module=self.norm1_context)
        else:
            raise ValueError(f'Please check JointRoPETransformerBlock or JointTransformerBlock norm1_context type: {type(self.norm1)}')
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        joint_attention_kwargs = joint_attention_kwargs or {}
        expand_gate_dim=len(hidden_states.shape)!=len(temb.shape)
        
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states=hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states=hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(hidden_states=encoder_hidden_states, emb=temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                hidden_states=encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output if expand_gate_dim else gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2 if expand_gate_dim else gate_msa2 * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        
        if expand_gate_dim:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        else:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = diffusers_attention._chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output if expand_gate_dim else gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output if expand_gate_dim else c_gate_msa * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            if expand_gate_dim:
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            else:
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
                
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = diffusers_attention._chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            
            if expand_gate_dim:
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            else:
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
                
        return encoder_hidden_states, hidden_states
    
class JointRoPETransformerBlock(JointTransformerBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only = False, qk_norm = None, use_dual_attention = False):
        super().__init__(dim, num_attention_heads, attention_head_dim, context_pre_only, qk_norm, use_dual_attention)

        self.attn.processor=JointAttnRopeProcessor()
        if use_dual_attention:
            self.attn2.processor=JointAttnRopeProcessor()

class JointAttnRopeProcessor(JointAttnProcessor2_0):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        hd_ids: Optional[torch.Tensor] = None,
        encoder_hd_ids: Optional[torch.Tensor] = None,
        rope_embed: FluxPosEmbed = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            
            if rope_embed is not None:
                assert hd_ids is not None and encoder_hd_ids is not None, ValueError('Please give the hidden states ids and encoder hidden states ids when process attention module!')
                hd_ids=torch.cat([hd_ids,encoder_hd_ids],dim=0)
        
        if rope_embed is not None:
            assert hd_ids is not None, ValueError('Please give the hidden states ids when process attention module!')
            rotary_embed=tuple(i.to(hd_ids.dtype) for i in rope_embed(hd_ids))
            query = apply_rotary_emb(query, rotary_embed)
            key = apply_rotary_emb(key, rotary_embed)
            
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class JointConditionAttnProcessor(JointAttnProcessor2_0):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        condition_hidden_states: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_len=encoder_hidden_states.shape[1]
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
        
        # condition projections.
        if condition_hidden_states is not None:
            condition_token_len=condition_hidden_states.shape[1]
            # coindition_hidden_states_query_proj = attn.condition_q_proj(condition_hidden_states)
            coindition_hidden_states_key_proj = attn.condition_k_proj(condition_hidden_states)
            coindition_hidden_states_value_proj = attn.condition_v_proj(condition_hidden_states)

            # coindition_hidden_states_query_proj = coindition_hidden_states_query_proj.view(
            #     batch_size, -1, attn.heads, head_dim
            # ).transpose(1, 2)
            coindition_hidden_states_key_proj = coindition_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            coindition_hidden_states_value_proj = coindition_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # if attn.condition_q_norm is not None:
            #     coindition_hidden_states_query_proj = attn.condition_q_norm(coindition_hidden_states_query_proj)
            if attn.condition_k_norm is not None:
                coindition_hidden_states_key_proj = attn.condition_k_norm(coindition_hidden_states_key_proj)

            # query = torch.cat([query, coindition_hidden_states_query_proj], dim=2)
            key = torch.cat([key, coindition_hidden_states_key_proj], dim=2)
            value = torch.cat([value, coindition_hidden_states_value_proj], dim=2)
            
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
                
            if condition_hidden_states is not None and hasattr(attn,'condition_q_proj'):
                encoder_hidden_states=encoder_hidden_states[:,:-condition_token_len]
            
            assert encoder_hidden_states.shape[1] == encoder_hidden_states_len, ValueError(f'Please check encoder hidden states length, original token len: {encoder_hidden_states_len}, now token len: {encoder_hidden_states.shape[1]}')
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states[:, : residual.shape[1]])
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


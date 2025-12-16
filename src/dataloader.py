import logging
import torch
import io
logger = logging.getLogger(__name__)
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
import os,json,random
import cv2
from glob import glob
import numpy as np
from src.UniGenUtils import debug_print

class MultiGen(Dataset):
    def __init__(self,args,vae_scale_factor,split="train",split_fre=0.8):
        super().__init__()
        self.data_base=args.dataset_name
        self.condition_types=args.condition_types
        self.resolution=args.resolution
        
        self.data = []
        for task in self.condition_types:
            json_file=os.path.join(self.data_base,'json_files','aesthetics_plus_all_group_'+task+'_all.json')

            each_task_data=[]
            with open(json_file,'rt') as f:
                for line in f:
                    each_task_data.append(json.loads(line))
            if split=='train':
                self.data.extend(each_task_data[:int(len(each_task_data)*split_fre)])
            else:
                self.data.extend(each_task_data[int(len(each_task_data)*split_fre):])
            
        self.image_processor=VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2 ,do_resize=True,do_convert_rgb=True)
    
    def resize_image_control(self, control_image, resolution):
        H, W, C = control_image.shape
        if W >= H:
            crop = H
            crop_l = random.randint(0, W-crop) # 2nd value is inclusive
            crop_r = crop_l + crop
            crop_t = 0
            crop_b = H
        else:
            crop = W
            crop_t = random.randint(0, H-crop) # 2nd value is inclusive
            crop_b = crop_t + crop
            crop_l = 0
            crop_r = W
        control_image = control_image[ crop_t: crop_b, crop_l:crop_r]
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(control_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img, [crop_t/H, crop_b/H, crop_l/W, crop_r/W]
    
    def resize_image_target(self, target_image, resolution, sizes):
        H, W, C = target_image.shape
        crop_t_rate, crop_b_rate, crop_l_rate, crop_r_rate = sizes[0], sizes[1], sizes[2], sizes[3]
        crop_t, crop_b, crop_l, crop_r = int(crop_t_rate*H), int(crop_b_rate*H), int(crop_l_rate*W), int(crop_r_rate*W)
        target_image = target_image[ crop_t: crop_b, crop_l:crop_r]
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(target_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img
        
    def __len__(self):
        return len(self.data)

    def collect_fun(self,batchs):
        pixel_values=torch.cat([batch['target_img'] for batch in batchs],dim=0)
        pixel_values=pixel_values.to(memory_format=torch.contiguous_format).float()
        condition_latents=torch.cat([batch['condition_img'] for batch in batchs],dim=0)
        condition_latents = condition_latents.to(memory_format=torch.contiguous_format).float()
        
        descriptions = [batch["prompt"] for batch in batchs]
        task_name = [batch["task"] for batch in batchs]
        return {"pixel_values": pixel_values, "condition_latents": condition_latents,
                "task_name":task_name,"descriptions": descriptions}

    def __getitem__(self, idx):
        idx=idx if idx < len(self.data) else random.randint(0,len(self.data)-1)
        item = self.data[idx]
        source_key=[key for key in item.keys() if 'control' in key]
        while len(source_key)!=1:
            return self.__getitem__(random.randint(0,len(self.data)-1))
        source_key=source_key[-1]
        
        source_filename = item[source_key]
        source_img_file=f'{self.data_base}/conditions/group_{source_filename.split("_group_")[-1]}'
        
        target_filename = item['source']
        if "./" == target_filename[0:2]:
            target_filename = target_filename[2:]
        target_img_file=f'{self.data_base}/images/{target_filename}'
        
        while not os.path.exists(source_img_file) or not os.path.exists(target_img_file):
            print(f'target image file: {target_img_file}, source image file: {source_img_file}')
            self.data.pop(idx)
            return self.__getitem__(random.randint(0,len(self.data)-1))
        
        prompt = item['prompt']
        
        source_img=cv2.imread(source_img_file)
        target_img=cv2.imread(target_img_file)
        
        source_img,  sizes = self.resize_image_control(source_img, self.resolution)
        target_img = self.resize_image_target(target_img, self.resolution, sizes)
        
        target_img=self.image_processor.preprocess(Image.fromarray(target_img).convert("RGB"),width=self.resolution,height=self.resolution)
        source_img=self.image_processor.preprocess(Image.fromarray(source_img).convert("RGB"),width=self.resolution,height=self.resolution)
        
        # prompt = prompt if random.uniform(0, 1) > 0.3 else '' # dropout rate 30%
        return dict(target_img=target_img, prompt=prompt, condition_img=source_img, task=source_key.replace("control_",""))

def get_dataset(args,split='train'):
    dataset = []
    assert isinstance(args.dataset_name,list),"dataset dir should be a list"
    if args.dataset_name is not None:
        for name in args.dataset_name:
            # Downloading and loading a dataset from the hub.
            dataset.append(load_dataset('parquet',data_dir=name,cache_dir=args.cache_dir,split='train'))
    dataset = concatenate_datasets(dataset)
    return dataset

class Subjects200K(Dataset):
    def __init__(self, base_path,condition_types,vae_scale_factor=None,resolution=512,split="train",test_split='depth_subject_pose.txt',max_data_len=None,image_processor=None):
        assert split in ['train','test'],ValueError('Check split, must in [train, test]')
        self.split=split
        self.resolution=resolution
        self.data=[]
        condition_types=condition_types if isinstance(condition_types,(list,tuple)) else [condition_types]
        
        test_base_imgs=[]
        for line in open(f'{base_path}/test_infos/{test_split}','r').readlines():
            test_base_imgs.append('/'.join(line.replace('\n','').split('/')[-2:]))
        self.test_base_imgs=test_base_imgs
        
        if split=='train':
            for task_name in condition_types:
                assert task_name in ['canny','depth','subject','openpose']
                if task_name=='depth':
                    self.data.extend(glob(f'{base_path}/score_*/*_depth_large_*.jpg'))
                elif task_name=='canny':
                    self.data.extend(glob(f'{base_path}/score_*/*_target_*.jpg'))
                else:
                    self.data.extend(glob(f'{base_path}/score_*/*_{task_name}_*.jpg'))
            if max_data_len is not None:
                self.data=random.sample(self.data,min(max_data_len,len(self.data)))
        else:        
            print(f'load test image info length: {len(test_base_imgs)}')
            for task_name in condition_types:
                for test_path in self.test_base_imgs:
                    if task_name=='depth':
                        self.data.extend(glob(f'{base_path}/{test_path.replace("_source_","_depth_large_")}'))
                    elif task_name=='canny':
                        self.data.extend(glob(f'{base_path}/{test_path.replace("_source_","_target_")}'))
                    else:
                        self.data.extend(glob(f'{base_path}/{test_path.replace("_source_",f"_{task_name}_")}'))
        
        if image_processor is not None:
            self.image_processor=image_processor
            debug_print(f'Use custom image processor: {self.image_processor}')
        elif vae_scale_factor is not None:
            self.image_processor=VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2 ,do_resize=True,do_convert_rgb=True)
            debug_print(f'Use vae image processor: {self.image_processor}')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        condition_path=self.data[idx]
        condition_img=None
        
        if '_depth_large_' in os.path.basename(condition_path):
            task_name='depth'
            target_path=condition_path.replace('_depth_large_','_target_')
        elif '_target_' in os.path.basename(condition_path):
            task_name='canny'
            target_path=condition_path
            condition_img=cv2.Canny(cv2.imread(condition_path),100,200)
        elif '_subject_' in os.path.basename(condition_path):
            task_name='subject'
            target_path=condition_path.replace('_subject_','_target_')
        elif '_openpose_' in os.path.basename(condition_path):
            task_name='openpose'
            target_path=condition_path.replace('_openpose_','_target_')
        else:
            raise ValueError(f'Unmatch task info from: {condition_path}')
        
        if self.split=='train':
            if '/'.join(target_path.split('/')[-2:]).replace('_target_','_source_') in self.test_base_imgs:
                return self.__getitem__(random.randint(0,len(self.data)-1))
        
        description_path=target_path.replace('_target_','_description_').replace(".jpg",'.json')
        with open(description_path,'r') as f:
            description=json.load(f)
        
        prompt = description['description_0']
        if prompt is not None:
            prompt=prompt.replace('<|endoftext|>','').replace("!","")
        else:
            prompt=""
        
        if getattr(self,'image_processor',None) is None:
            # for Unicontrol
            source_img,  sizes = self.resize_image_control(condition_img if condition_img is not None else cv2.imread(condition_path), self.resolution)
            target_img = self.resize_image_target(cv2.imread(target_path), self.resolution, sizes)
            
            # Do not forget that OpenCV read images in BGR order.
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

            # Normalize source images to [0, 1].
            source_img = source_img.astype(np.float32) / 255.0

            # Normalize target images to [-1, 1]. 
            if self.split!='test':
                target_img = (target_img.astype(np.float32) / 127.5) - 1.0
                prompt = prompt if random.uniform(0, 1) > 0.3 else '' # dropout rate 30%
            return dict(jpg=target_img, txt=prompt, hint=source_img, task=f'control_{task_name}')
        else:
            target_img=Image.open(target_path).convert("RGB")
            condition_img=Image.fromarray(cv2.cvtColor(condition_img,cv2.COLOR_BGR2RGB)) if condition_img is not None else Image.open(condition_path).convert("RGB")
        
            if self.split=='test':
                return dict(target_img=target_img.convert("RGB"),condition_img=condition_img.convert("RGB"),task=task_name,id=idx,prompt=prompt)
            
            target_img=self.image_processor.preprocess(target_img.convert("RGB"),width=self.resolution,height=self.resolution)
            condition_img=self.image_processor.preprocess(condition_img.convert("RGB"),width=self.resolution,height=self.resolution)
            
            # dropout rate 30%
            return dict(target_img=target_img, prompt=prompt if random.uniform(0, 1) > 0.3 else '', condition_img=condition_img, task=task_name, id=idx)

def collate_fn(batchs,split='train'):
    rtn_batch=dict(batch_ids=[],descriptions=[],task_names=[])
    for batch in batchs:
        rtn_batch['descriptions'].append(batch['prompt'])
        rtn_batch['task_names'].append(batch['task'])
        rtn_batch['batch_ids'].append(batch['id'])
        if split=='train':
            rtn_batch.setdefault('pixel_values',[])
            rtn_batch['pixel_values'].append(batch['target_img'])
            
            rtn_batch.setdefault('condition_latents',[])
            rtn_batch['condition_latents'].append(batch['condition_img'])

        else:
            rtn_batch.setdefault('target_img',[])
            rtn_batch['target_img'].append(batch['target_img'])
            
            rtn_batch.setdefault('condition_img',[])
            rtn_batch['condition_img'].append(batch['condition_img'])
            
        if 'mask_img' in batch.keys():
            rtn_batch.setdefault('mask_img',[])
            rtn_batch['mask_img'].append(batch['mask_img'])
            
        if 'mask' in batch.keys():
            rtn_batch.setdefault('mask',[])
            rtn_batch['mask'].append(batch['mask'])
            
        if 'mask_prompt' in batch.keys():
            rtn_batch.setdefault('mask_prompt',[])
            rtn_batch['mask_prompt'].append(batch['mask_prompt'])
    
    if 'pixel_values' in rtn_batch.keys():
        rtn_batch['pixel_values']=torch.cat(rtn_batch['pixel_values'],dim=0).to(memory_format=torch.contiguous_format).float()
    
    if 'condition_latents' in rtn_batch.keys() and split=='train':
        rtn_batch['condition_latents']=torch.cat(rtn_batch['condition_latents'],dim=0).to(memory_format=torch.contiguous_format).float()
    
    if 'mask_img' in rtn_batch.keys() and split=='train':
        rtn_batch['mask_img']=torch.cat(rtn_batch['mask_img'],dim=0).to(memory_format=torch.contiguous_format).float()
    
    if 'mask' in rtn_batch.keys():
        rtn_batch['mask']=torch.stack(rtn_batch['mask'],dim=0).to(memory_format=torch.contiguous_format).float()

    return rtn_batch


class MultiConditionSubjects200K(Dataset):
    def __init__(self, base_path,condition_types,vae_scale_factor=None,resolution=512,split="train",test_split='depth_subject_pose.txt',max_data_len=None):
        assert split in ['train','test'],ValueError('Check split, must in [train, test]')
        self.split=split
        self.resolution=resolution
        self.data=[]

        self.condition_map=dict(
            depth='_depth_large_',
            canny='_target_',
            subject='_subject_',
            openpose='_openpose_'
        )
        
        test_base_imgs=[]
        for line in open(f'{base_path}/test_infos/{test_split}','r').readlines():
            test_base_imgs.append('/'.join(line.replace('\n','').split('/')[-2:]))
        self.test_base_imgs=test_base_imgs
        
        if split=='train':
            self.data=glob(f'{base_path}/score_*/*_target_*.jpg')
            print(f'load train image info length: {len(self.data)}')
            if 'openpose' in condition_types:
                for path in self.data:
                    if not os.path.exists(path.replace('_target_',"_openpose_")):
                        self.data.remove(path)
                print(f'filter openpose image length: {len(self.data)}')
        else:
            print(f'load test image info length: {len(test_base_imgs)}')
            for test_path in self.test_base_imgs:
                multi_condition_info=dict()
                for cond_type in condition_types:
                    load_path=glob(f'{base_path}/{test_path.replace("_source_",self.condition_map[cond_type])}')
                    if len(load_path)==0:
                        print(f'load {cond_type} error, not found file in path: {base_path}/{test_path.replace("_source_","_depth_large_")}')
                        break
                    multi_condition_info[cond_type]=random.sample(load_path,1)[0]
                load_path=glob(f'{base_path}/{test_path.replace("_source_","_target_")}')
                if len(load_path)==0:
                    print(f'load target image error, not found file in path: {base_path}/{test_path.replace("_source_","_target_")}')
                    break
                multi_condition_info['target']=random.sample(load_path,1)[0]
                self.data.append(multi_condition_info)
            
        if vae_scale_factor is not None:
            self.image_processor=VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2 ,do_resize=True,do_convert_rgb=True)

        assert hasattr(self,'image_processor'),'image_processor must be defined, please give vae_scale_factor'
        self.condition_types=condition_types
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx>=len(self.data):
            idx=random.randint(0,len(self.data)-1)

        data,rtn_infos=self.data[idx],dict()
        if self.split!='train':
            for condition_type,condition_path in data.items():
                if condition_type=='canny':
                    rtn_infos[condition_type]=Image.fromarray(cv2.cvtColor(cv2.Canny(cv2.imread(condition_path),100,200),cv2.COLOR_BGR2RGB)) 
                else:
                    rtn_infos[condition_type]=Image.open(condition_path).convert("RGB")
            
            description_path=data['target'].replace('_target_','_description_').replace(".jpg",'.json')
        else:
            if '/'.join(data.split('/')[-2:]).replace('_target_','_source_') in self.test_base_imgs:
                return self.__getitem__(random.randint(0,len(self.data)-1))
        
            for condition_type in self.condition_types:
                if condition_type=='canny':
                    condition_img=cv2.Canny(cv2.imread(data),100,200)
                    condition_img=Image.fromarray(cv2.cvtColor(condition_img,cv2.COLOR_BGR2RGB))
                else:
                    if not os.path.exists(data.replace('_target_',self.condition_map[condition_type])):
                        print(f'not found {condition_type} image in path: {data.replace("_target_",self.condition_map[condition_type])}, will continue')
                        return self.__getitem__(random.randint(0,len(self.data)-1))
                    condition_img=Image.open(data.replace('_target_',self.condition_map[condition_type])).convert("RGB")

                condition_img=self.image_processor.preprocess(condition_img.convert("RGB"),width=self.resolution,height=self.resolution)
            
                rtn_infos[condition_type]=condition_img
            rtn_infos['target']=self.image_processor.preprocess(Image.open(data).convert("RGB"),width=self.resolution,height=self.resolution)
            description_path=data.replace('_target_','_description_').replace(".jpg",'.json')

        with open(description_path,'r') as f:
            description=json.load(f)
        
        prompt = description['description_0']
        if prompt is not None:
            prompt=prompt.replace('<|endoftext|>','').replace("!","")
        else:
            prompt=""

        if self.split=='train' and random.uniform(0, 1) <= 0.3:
            prompt=""

        rtn_infos['prompt']=prompt
        
        rtn_infos['id']=idx
        return rtn_infos

def collect_multi_condition_fun(batchs,condition_types,split='train'):
    rtn_batch=dict(batch_ids=[],target_img=[],descriptions=[])
    for batch in batchs:
        rtn_batch['descriptions'].append(batch.pop('prompt'))
        rtn_batch['batch_ids'].append(batch.pop('id'))
  
        if split!='train':
            rtn_batch['target_img'].append(batch.pop('target'))
        else:
            rtn_batch.setdefault('pixel_values',[]).append(batch['target'])
            
        for condition_type in condition_types:
            rtn_batch.setdefault(condition_type,[]).append(batch.pop(condition_type))
    
    if split=='train':
        rtn_batch['pixel_values']=torch.cat(rtn_batch['pixel_values'],dim=0).to(memory_format=torch.contiguous_format).float()

        for condition_type in condition_types:
            rtn_batch[condition_type]=torch.cat(rtn_batch[condition_type],dim=0).to(memory_format=torch.contiguous_format).float()

    return rtn_batch

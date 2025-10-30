import copy
import json
import yaml
import math
import random
import os
import transformers
import torch
import megfile
import PIL
import os
import random
import megfile
import rllava.utils.torch_functional as VF
from typing import Optional, List, Any
from qwen_vl_utils import smart_resize
from dataclasses import dataclass
from jinja2 import Template
from typing import Dict,  Sequence
from PIL import Image, ImageFile
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from rllava.data.data_utils import process_image, process_video
from rllava.model.patch.qwen2_vl import get_rope_index
from .template import TemplateFactory
from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from rllava.utils.arguments import DataArguments
from rllava.utils.constants import *



ImageFile.LOAD_TRUNCATED_IMAGES = True


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # Local directory containing dataset (e.g., HF cache format)
            from datasets import load_from_disk
            try:
                # Try to load as a DatasetDict with splits
                full_dataset = load_from_disk(data_path)
                if hasattr(full_dataset, 'keys') and data_split in full_dataset:
                    self.dataset = full_dataset[data_split]
                else:
                    # It's a single dataset, use it directly
                    self.dataset = full_dataset
            except:
                # Fallback: try loading as Arrow dataset
                self.dataset = load_dataset("arrow", data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            # Single file
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # Remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            # self.dataset = self.dataset.filter(
            #     self._filter_overlong_prompts,
            #     desc="Filtering overlong prompts",
            #     num_proc=filter_overlong_prompts_workers,
            # )
            self.dataset = self.dataset.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=filter_overlong_prompts_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataset)}")

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example) # [{'role': 'user', 'content': [{...}, {...}]}]
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            # Handle both single image and list of images #mzh
            if not isinstance(images, list): #mzh
                images = [images] #mzh
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            # image_grid_thw = model_inputs.pop("image_grid_thw")[0]
            # pixel_values = model_inputs.pop("pixel_values")
            example["multi_modal_data"] = {"images": images}
            # example['pixel_values'] = pixel_values
            # example['image_grid_thw'] = image_grid_thw
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from rllava.model.patch.qwen3_vl import get_rope_index
            else:
                from rllava.model.patch.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
            # position_ids = vision_position_ids
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example


class ReasoningDataset(Dataset):
    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.problem_key = self.data_args.problem_key
        self.answer_key = self.data_args.answer_key
        self.image_key = self.data_args.image_key
        self.image_dir = self.data_args.image_dir
        self.template = TemplateFactory(data_args.conv_version)(data_args.system_prompt_template, 
                                                                data_args.question_template, 
                                                                data_args.answer_template)

        data_path = data_args.data_path
        if data_path.endswith(".yaml"):
            self.list_data_dict = self.get_datas_from_yaml(data_path)
        elif data_path.endswith(".json"):
            with open(data_path, "r") as file:
                self.list_data_dict = json.load(file)
            print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        else:
            try:
                # huggingface dataset
                hf_dataset = load_dataset(data_path)
                if 'train' in hf_dataset:
                    split = 'train'
                else:
                    split = list(hf_dataset.keys())[0]
                if data_args.train_sample_size is not None:
                    self.list_data_dict = hf_dataset[split].select(range(data_args.train_sample_size)).to_list()
                else:
                    self.list_data_dict = hf_dataset[split].to_list()
                print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
            except Exception as e:
                raise ValueError(f"Unsupported file type: {data_path}")
            
        if data_args.train_sample_size is not None:
            self.list_data_dict = self.list_data_dict[:data_args.train_sample_size]
            
    def get_datas_from_yaml(self, yaml_path):
        '''
        file should be in the format of:
        datasets:
            - json_path: xxxx1.json
            sampling_strategy: first:1000
            - json_path: xxxx2.json
            sampling_strategy: end:3000
            - json_path: xxxx3.json
            sampling_strategy: random:999
        '''
        datas_from_yaml = []
        with open(yaml_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")

            for data in datasets:
                json_path = data.get("json_path")
                sampling_strategy = data.get("sampling_strategy", "all")
                sampling_number = None

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]
                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                datas_from_yaml.extend(cur_data_dict)
        return datas_from_yaml

    def __len__(self):
        return len(self.list_data_dict)
    
    def load_image(self, image_path):
        # when image is on oss, please run `unset http_proxy https_proxy all_proxy no_proxy`
        from io import BytesIO
        os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
        if isinstance(image_path, bytes):
            image = Image.open(BytesIO(image_path), "r").convert('RGB')
        elif 's3://' in image_path:
            with megfile.smart_open(image_path, "rb") as f:
                bytes_data = f.read()
            image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, i):
        image_key = self.data_args.image_key
        image_dir = self.data_args.image_dir

        example = self.list_data_dict[i]
        
        if image_key in example:
            # huggingface dataset -> image item is a PIL.Image.Image object
            if isinstance(example[image_key], str):
                image_path = os.path.join(image_dir, example[image_key])
                os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
                # In case the image is not found
                while not megfile.smart_exists(image_path):
                    print(f"Warning: Image {image_path} not found, randomly selecting another image")
                    new_index = random.randint(0, len(self.list_data_dict)-1)
                    example = self.list_data_dict[new_index]
                    image_path = os.path.join(image_dir, example[image_key])
            else:
                image_path = example[image_key]['bytes']
            image = self.load_image(image_path)

            image_size = self.data_args.image_size
            if image_size is not None:
                image_size = tuple(map(int, image_size.split()))
                image = image.resize(image_size)
            else:
                width, height = image.size
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=28,
                    min_pixels=self.data_args.min_pixels,
                    max_pixels=self.data_args.max_pixels,
                )
                image = image.resize((resized_width, resized_height))
        else:
            image = None
        return self.template.make_inputs(example, 
                                         image,
                                         self.problem_key, 
                                         self.answer_key)
        
    
class ReasoningSFTDataset(ReasoningDataset):
    def __getitem__(self, i):
        example = super().__getitem__(i)
        prompt = json.loads(example['prompt'])
        prompt.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(self.template.answer.format(answer=example['solution']))}
            ]
        })
        return {
            'image': example['image'],
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': json.dumps(prompt),
        }

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

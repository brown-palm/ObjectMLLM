import torch
import transformers
import argparse
import random
import json
import sys
sys.path.append('./')
import os
from tqdm import tqdm

from videollama2.conversation import conv_templates
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_video_segment
from videollama2.model.builder import load_pretrained_model

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_frames", type=int, default=8)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--finetuned_path", type=str, default=None)
parser.add_argument("--video_folder", type=str, default=None)

args = parser.parse_args()

num_frames = args.num_frames
if args.finetuned_path:
    model_path = args.finetuned_path
elif num_frames == 8:
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
elif num_frames == 16:
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-16F'

if 'DAMO-NLP-SG' in model_path:
    model_name = get_model_name_from_path(model_path)
else:
    model_name = 'videollama'
    if os.path.exists(os.path.join(model_path, 'adapter_model.bin')):
        model_name = model_name + '_lora'

tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
model = model.to('cuda:0')
conv_mode = 'llama_2'

if args.finetuned_path:
    logdir = args.finetuned_path
else:
    logdir = f'experiments/{args.dataset}_{model_name}_Frame{num_frames}'

def response_to_number(x, n_choices):
    if x[0] == '(':
        x = x[1:]
    x = ord(x[0]) - ord('A')
    if x < 0 or x >= n_choices:
        x = None
    return x

def process(video_path, start, end, question):
    # Visual preprocess (load & transform image or video).
    tensor = process_video_segment(video_path, start, end, processor, model.config.image_aspect_ratio, num_frames=num_frames)# .to(dtype=torch.float16, device='cuda', non_blocking=True)
    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]

    # Text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt')

    return tensor, input_ids

class MCQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_folder):
        super().__init__()
        fn = {
            'clevrer_vis_box': './data/clevrer/vis_box_eval.json',
            'ptest_vis_box': './data/ptest/vis_box_eval.json',
            'star_vis_box': './data/star/vis_box_eval.json',
            'nextqa_vis_box': './data/nextqa/vis_box_eval.json',
            'intentqa_vis_box': './data/intentqa/vis_box_eval.json'
        }[dataset]
        self.data = json.load(open(fn))
        self.video_folder = video_folder

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        video_path = os.path.join(self.video_folder, sample['video_path'])
        start, end = None, None
        n_c = len(sample['options'])
        question = f'Please provide a single-letter answer (from A to {chr(ord("A") + n_c - 1)}) to the following multiple-choice question, and your answer must be one of the letters from A to {chr(ord("A") + n_c - 1)}. You must not provide any other response or explanation. If you are not sure, answer with the most likely answer.\n Here is the question: {sample["question"]}\n Here are the choices:\n'
        for i, opt in enumerate(sample['options']):
            question = question + f'({chr(ord("A") + i)}) {opt}.\n'
        tensor, input_ids = process(video_path, start, end, question)
        return tensor, input_ids, sample['answer'], n_c, str(sample['video_id']) + '_' + str(sample['question_id'])

dataset = MCQADataset(args.dataset, args.video_folder)
dataloader = DataLoader(dataset, 1, shuffle=False, drop_last=False, num_workers=args.num_workers)

cnt = 0
cor = 0
invalid = 0
preds = {}

for sample in tqdm(dataloader):
    tensor, input_ids, label, n_c, idx = sample
    tensor = [tensor[0].to(dtype=torch.float16, device='cuda', non_blocking=True)]
    input_ids = input_ids.to('cuda:0')
    idx = idx[0]
    n_c = n_c[0].item()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=['video'],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=3,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    pred = response_to_number(output, n_c)
    if pred is None:
        invalid += 1
        pred = random.randint(0, n_c - 1)
    cnt += 1
    if pred == label[0].item():
        cor += 1
    preds[idx] = pred


os.makedirs(logdir, exist_ok=True)
with open(f'{logdir}/results.json', 'w') as f:
    f.write(json.dumps({
        'Total number of questions': cnt,
        'Number of correctly answered questions': cor,
        'Accuracy': cor / cnt,
        'Number of invalid responses': invalid
    }, indent=4))
with open(f'{logdir}/predictions.json', 'w') as f:
    f.write(json.dumps(preds, indent=4))
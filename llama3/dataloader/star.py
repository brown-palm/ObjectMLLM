import torch
from .base_dataset import BaseDataset, video_loader, box_loader
import pandas as pd
import json

from collections import defaultdict

class STAR(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = json.load(open(f'./data/star/STAR_{split}.json'))
        if args.use_cap:
            self.caption = json.load(open(f'./data/star/caption_{args.cap_model}.json'))
        self.mm_features = {}
        self.mm_texts = {}
        self.mm_used = args.mm_used
        self.max_feats = args.max_feats
        if args.use_vis:
            self.mm_texts['video'] = defaultdict(lambda: "Video:<|video|>")
            self.mm_features['video'] = video_loader(f'./data/star/clipvitl14.pth', self.max_feats['video'])
        if args.use_box:
            box_folder = './data/star/bbox'
            data = json.load(open(f'./data/star/bbox_{args.box_format}.json'))
            self.mm_texts['box'] = {k: v['text'] for k, v in data.items()}
            self.mm_features['box'] = box_loader(
                box_folder = box_folder,
                _format = args.box_format,
                data = data
            )
            
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.num_options = 4
        self.use_cap = args.use_cap
        self.use_box = args.use_box
        print("Use caption", args.use_cap)
        print("Used modalities:", self.mm_used)
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx):
        question = self.data[idx]['question'].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [x['choice'] for x in self.data[idx]['choices']]
        answer = options.index(self.data[idx]['answer'])
        vid = self.data[idx]['video_id']
        qid = self.data[idx]['question_id']
        qid = str(vid) + "_" + str(qid)

        caption = ""
        for m in self.mm_used:
            caption = caption + self.mm_texts[m][vid] + '\n'
        if self.use_cap:
            caption = caption + self.caption[vid].replace('In the video we see these scenes', 'This video shows') + '\n'
            
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "
        text = {'c_text': caption, 'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text, answer

    def __getitem__(self, idx):
        vid = self.data[idx]['video_id']
        text, answer = self._get_text(idx)
        text_id, label, mm_starts, label_mask = self._get_text_token(text, answer)
        mm_features = {}
        for m in self.mm_used:
            mm_features[m] = [torch.Tensor(x) for x in self.mm_features[m][vid]]
        
        return {"vid": vid, "mm_features": mm_features, "mm_starts": mm_starts, "text": text, "text_id": text_id, "label": label,
                "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": -1}

    def __len__(self):
        return len(self.data)
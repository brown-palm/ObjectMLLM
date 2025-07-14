import torch
from .base_dataset import BaseDataset, box_loader, video_loader
import pandas as pd
import json

from collections import defaultdict

class CLEVRER(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f'./data/clevrer/{split}.csv')
        if args.use_cap:
            self.caption = json.load(open(f'./data/clevrer/caption_{args.cap_model}.json'))
        self.mm_features = {}
        self.mm_texts = {}
        self.mm_used = args.mm_used
        self.max_feats = args.max_feats
        if args.use_vis:
            self.mm_texts['video'] = defaultdict(lambda: "Video:<|video|>")
            self.mm_features['video'] = video_loader(f'./data/clevrer/clipvitl14.pth', self.max_feats['video'])
        if args.use_box:
            box_folder = './data/clevrer/bbox'
            data = json.load(open(f'./data/clevrer/bbox_{args.box_format}.json'))
            self.mm_texts['box'] = {k: v['text'] for k, v in data.items()}
            self.mm_features['box'] = box_loader(
                box_folder = box_folder,
                _format = args.box_format,
                data = data
            )
            
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.use_cap = args.use_cap
        self.use_box = args.use_box
        print("Use caption", args.use_cap)
        print("Used modalities:", self.mm_used)
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx):
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        num_options = self.data['num_option'].values[idx]
        options = [self.data[f'a{i}'].values[idx] for i in range(num_options)]
        vid = self.data['video'].values[idx]
        qid = self.data['qid'].values[idx]

        caption = ""
        for m in self.mm_used:
            caption = caption + self.mm_texts[m][vid] + '\n'
        if self.use_cap:
            caption = caption + self.caption[vid] + '\n'

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "
        text = {'c_text': caption, 'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def __getitem__(self, idx):
        vid = self.data['video'].values[idx]
        answer = self.data['answer'].values[idx]
        text = self._get_text(idx)
        text_id, label, mm_starts, label_mask = self._get_text_token(text, answer)
        mm_features = {}
        for m in self.mm_used:
            mm_features[m] = [torch.Tensor(x) for x in self.mm_features[m][vid]]

        return {"vid": vid, "mm_features": mm_features, "mm_starts": mm_starts, "text": text, "text_id": text_id, "label": label,
                "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": -1}

    def __len__(self):
        return len(self.data)
import torch
from .base_dataset import BaseDataset, video_loader
import pandas as pd
import json

from collections import defaultdict

class BABEL(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = json.load(open(f'./data/babel/{split}.json'))
        if args.use_cap:
            self.caption = json.load(open(f'./data/babel/caption_{args.cap_model}.json'))
        self.mm_features = {}
        self.mm_texts = {}
        self.mm_used = args.mm_used
        self.max_feats = args.max_feats
        if args.use_vis:
            self.mm_texts['video'] = defaultdict(lambda: "Video:<|video|>")
            self.mm_features['video'] = video_loader(f'./data/babel/clipvitl14.pth', self.max_feats['video'])
        if args.use_pose:
            self.pose = json.load(open(f'./data/babel/pose_{args.pose_format}.json'))

        self.answer_dic = ['fall', 'golf', 'march', 'bow', 'kneel', 'throw', 'squat', 'right foot', 'flip', 'place something', 'yawn', 'catch', 'crawl', 'jumping jacks', 'skip', 'wave', 'walk', 'drink', 'dribble', 'right arm', 'cartwheel', 'left foot', 'flail arms', 'crouch', 'left leg', 'leap', 'backwards', 'shuffle', 'kick', 'juggle', 'jog', 'take/pick something up', 'right hand', 'left hand', 'sit', 'punch', 'jump rope', 'left arm', 'stand up', 'forward', 'knock', 'left', 'run', 'tie', 'right', 'lunge', 'jump', 'clap', 'duck', 'right leg']
        self.answer_mapping = {i: k for i, k in enumerate(self.answer_dic)}
        self.num_options = 50
        self.use_cap = args.use_cap
        self.use_pose = args.use_pose
        print("Use caption", args.use_cap)
        print("Use pose", args.use_pose)
        print("Used modalities:", self.mm_used)
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx):
        question = self.data[idx]['question'].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        answer = self.answer_dic.index(self.data[idx]['answer'])
        vid = self.data[idx]['babel_id']
        qid = self.data[idx]['question_id']
        qid = str(vid) + "_" + str(qid)

        caption = ""
        for m in self.mm_used:
            caption = caption + self.mm_texts[m][vid] + '\n'
        if self.use_pose:
            caption = caption + self.pose[vid] + '\n'
        if self.use_cap:
            caption = caption + self.caption[vid].replace('In the video we see these scenes', 'This video shows') + '\n'
        q_text = f"Question: {question}\n"
        o_text = ""
        
        a_text = "Answer: The answer is "
        text = {'c_text': caption, 'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': self.answer_dic}
        return text, answer

    def __getitem__(self, idx):
        vid = self.data[idx]['babel_id']
        text, answer = self._get_text(idx)
        text_id, label, mm_starts, label_mask = self._get_text_token(text, answer)
        mm_features = {}
        for m in self.mm_used:
            mm_features[m] = [torch.Tensor(x) for x in self.mm_features[m][vid]]
        
        return {"vid": vid, "mm_features": mm_features, "mm_starts": mm_starts, "text": text, "text_id": text_id, "label": label,
                "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": -1}

    def __len__(self):
        return len(self.data)
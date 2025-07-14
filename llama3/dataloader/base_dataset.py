import torch
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import numpy as np
from collections import defaultdict

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.features_dim = 768
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.split = split
    
    def _get_padding_id(self, text_id):
        padding_text_id = torch.zeros((len(text_id), self.max_seq_len), dtype=torch.int64) - 1
        for i, tid in enumerate(text_id):
            padding = self.max_seq_len - len(tid)
            if padding >= 0:
                padding_text_id[i, :len(tid)] = tid
            else:
                padding_text_id[i] = tid[:self.max_seq_len]
                print('max sequence length overflow')
        return padding_text_id
    
    def _get_text_token(self, text, answer):
        vqa_id, vqa_prefix_index, vqa_mm_starts = self.tokenizer.encode_vqa(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        vqa_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vqa_id]
        vqa_padding_text_id = self._get_padding_id(vqa_id)

        # label
        vqa_label = copy.deepcopy(vqa_padding_text_id)
        vqa_label[:, :vqa_prefix_index] = -1
        vqa_label_mask = vqa_label.ge(0)
        vqa_label[~vqa_label_mask] = 0
        vqa_label_mask = vqa_label_mask.float()  
                
        # text mask
        vqa_text_mask = vqa_padding_text_id.ge(0)
        vqa_padding_text_id[~vqa_text_mask] = 0
        
        text_id = {'vqa': vqa_padding_text_id}
        label = {'vqa': vqa_label}
        mm_starts = {'vqa': vqa_mm_starts}
        label_mask = {'vqa': vqa_label_mask}
        return text_id, label, mm_starts, label_mask

def box_loader_boxes(boxes, index):
    oid = index['object_id']
    frame_id = index['frame_id']
    box = boxes[oid]['bounding_boxes'][frame_id]
    return box[None]

def box_loader(box_folder, _format, data):
    if 'textual' in _format:
        return defaultdict(lambda: [])
    else:
        loader = box_loader_boxes

    ret = {}
    print('loading boxes')
    from glob import glob
    for vid, d in data.items():
        pths = glob(f'{box_folder}/{vid}/*.npz')
        boxes = {}
        for pth in pths:
            oid = pth.split('/')[-1].split('.')[0]
            boxes[int(oid)] = dict(np.load(pth))
        ret[vid] = [loader(boxes, x) for x in d['box']]
    print('boxes loaded')
    return ret

def video_loader(feature_fn, max_feats):
    print('loading video features')
    features = torch.load(feature_fn)
    ret = {}
    for video_id in features:
        video = features[video_id].float()
        if len(video) > max_feats:
            sampled = []
            for j in range(max_feats):
                sampled.append(video[(j * len(video)) // max_feats])
            video = torch.stack(sampled)
        elif len(video) < max_feats:
            video_len = len(video)
            features_dim = video.shape[-1]
            video = torch.cat([video, torch.zeros(max_feats - video_len, features_dim)], dim=0)
        ret[video_id] = [video]
    print('video features loaded')
    return ret
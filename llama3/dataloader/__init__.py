import torch
from util import misc
from .nextqa import NextQA
from .ptest import PTest
from .intentqa import IntentQA
from .clevrer import CLEVRER
from .star import STAR
from .babel import BABEL

dataset_mapping = {'nextqa': NextQA, 'intentqa': IntentQA, 'ptest': PTest, 'clevrer': CLEVRER, 'star': STAR, 'babel': BABEL}
num_options_mapping = {'nextqa': 5, 'intentqa': 5, 'ptest': 3, 'clevrer': -1, 'star': 4, 'babel': 50}

def load_data(args, tokenizer, split='train'):
    args.num_options = num_options_mapping[args.dataset]
    dataset = dataset_mapping[args.dataset](args=args, tokenizer=tokenizer, split=split)
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=batch_collate,
                                              pin_memory=args.pin_mem, drop_last=False)

    return data_loader

def batch_collate(batch):
    bs = len(batch)
    vid = [batch[i]["vid"] for i in range(bs)]
    mm_features = [batch[i]['mm_features'] for i in range(bs)]
    text = [batch[i]["text"] for i in range(bs)]
    qid = [batch[i]["qid"] for i in range(bs)]
    qtype = torch.tensor([batch[i]['qtype'] for i in range(bs)])
    
    vqa_id = torch.stack([batch[i]['text_id']['vqa'] for i in range(bs)])
    text_id = {'vqa': vqa_id}
    
    vqa_label = torch.stack([batch[i]['label']['vqa'] for i in range(bs)])
    label = {'vqa': vqa_label}
    
    vqa_mm_starts = [batch[i]["mm_starts"]['vqa'] for i in range(bs)]
    mm_starts = {'vqa': vqa_mm_starts}
    
    vqa_label_mask = torch.stack([batch[i]["label_mask"]['vqa'] for i in range(bs)])
    label_mask = {'vqa': vqa_label_mask}

    answer = torch.tensor([batch[i]["answer"] for i in range(bs)])

    return {"vid": vid, "mm_features": mm_features, "mm_starts": mm_starts, "text": text, "text_id": text_id, "label": label,
            "label_mask": label_mask, "qid": qid, "answer": answer, "qtype": qtype}
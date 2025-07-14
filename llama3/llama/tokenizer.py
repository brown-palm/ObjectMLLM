# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from .llama3_tokenizer import Tokenizer as Tokenizer3
from sentencepiece import SentencePieceProcessor

from logging import getLogger
from typing import List
import os
import torch

logger = getLogger()

mm_tokens = {
    '<|video|>': 'video',
    '<|box|>': 'box'
}

class Tokenizer_llama3:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.tk_model = Tokenizer3(model_path=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.tk_model.n_words
        self.bos_id: int = self.tk_model.bos_id
        self.eos_id: int = self.tk_model.eos_id
        self.pad_id: int = self.tk_model.pad_id
        self.unk_id: int = self.tk_model.pad_id
        
        self.v_token_id = 10955
        self.q_token_id = 14924
        self.a_token_id = 16533
        self.nl_id = 198
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tk_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_w_mm_tokens(self, s, max_feats):
        texts = [s]
        for m in mm_tokens:
            new_texts = []
            for t in texts:
                t = t.split(m)
                for x in t:
                    if x.strip() != '':
                        new_texts.append(x)
                    new_texts.append(m)
                new_texts = new_texts[:-1]
            texts = new_texts
        mm_starts = []
        L = [self.bos_id]
        for t in texts:
            if t in mm_tokens:
                m = mm_tokens[t]
                mm_starts.append((m, len(L)))
                L = L + [self.unk_id for _ in range(max_feats[m])]
            else:
                L = L + self.tk_model.encode(t)
        L = L + [self.eos_id]
        return L, mm_starts

    def encode_vqa(self, text=None, max_feats=None, split='train', answer_mapping=None, answer=None):
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        q_text = text['q_text'] # question
        o_text = text['o_text'] # options
        a_text = text['a_text'] # answer
        c_text = text['c_text'] # multimodal inputs

        if split == 'train':
            s = i_text + c_text + q_text + o_text + a_text + answer_mapping[answer]
            w, mm_starts = self.encode_w_mm_tokens(s, max_feats)
            t = [w]
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for k, v in answer_mapping.items():
                s = i_text + c_text + q_text + o_text + a_text + v
                w, mm_starts = self.encode_w_mm_tokens(s, max_feats)
                t.append(w)
            prefix_index = t[answer].index(self.a_token_id) + 5
        return t, prefix_index, mm_starts

    def decode(self, t: List[int]) -> str:
        return self.tk_model.decode(t)
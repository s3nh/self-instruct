import torch
from typing import List, Dict, Union
from typing import Any, TypeVar
from torch.utils.data import Dataset

class InstructDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        
        self.input_ids: List = []
        self.attn_masks: List = []
        self.labels: List = []
        for txt in data:
            instruction = generate_prompt(instruction = txt['instruction'], 
                                          input = txt['input'], 
                                          response = txt['output'])
            encodings_dict = tokenizer(instruction, padding="max_length", max_length = max_length, truncation=True)
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx) -> Tuple[List, List]:
        return self.input_ids[idx], self.attn_masks[idx]    

    

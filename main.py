import argparse
import pandas as pd
import pathlib
import torch
from pathlib import Path
from torch.utils.data import random_split
from transformers import TrainingArguments
from transformers import Trainer
from typing import List, Dict, Union
from typing import Any, TypeVar, Tuple

from self_instruct.cfg import BasicConfig, TrainConfig 
from self_instruct.utils import parse_model_name
from self_instruct.utils import load_model
from self_instruct.utils import load_tokenizer
from self_instruct.utils import create_outpath
from self_instruct.dataset import InstructDataset

Pathable = Union[str, pathlib.Path]
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs="?", default="")
    parser.add_argument('-t', '--tokenizer', nargs="?", default="")
    parser.add_argument('-l', '--lora',  nargs="?", default=False)
    args = parser.parse_args()
    return args

def  prepare_dataset(input_path: Pathable, **kwargs) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    data = pd.read_csv(input_path)
    dict_data = pd.DataFrame.to_dict(data, orient="records")
    dataset  = InstructDataset(data = dict_data, tokenizer = kwargs['tokenizer'], max_length = kwargs['config'].max_length)
    return split_data(dataset = dataset, ratio = kwargs['config'].split_ratio)

def split_data(dataset: torch.utils.data.Dataset, ratio: float = 0.9):
    train_size: int = int(ratio * len(dataset))
    val_size: int = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def main():
    config = BasicConfig()
    trconfig = TrainConfig()
    args = get_args()
    model_name = args.model if args.model is not "" else config.model_name
    tokenizer_name = args.tokenizer if args.tokenizer is not "" else config.tokenizer_name
    parsed_model_name = parse_model_name(model_name)
    parsed_tokenizer_name = parse_model_name(tokenizer_name) 
    model = load_model(model_name)
    tokenizer = load_tokenizer(tokenizer_name)
    train_dataset, val_dataset = prepare_dataset(input_path = config.dataset, tokenizer = tokenizer, config = config)
    
    #Create output path
    output_path = create_outpath(model_name = model_name)
    
        
    training_args = TrainingArguments(output_dir = output_path, 
                                      num_train_epochs = trconfig.n_epochs, 
                                      logging_steps = trconfig.logging_steps, 
                                      per_device_train_batch_size = trconfig.train_batch_size, 
                                      per_device_eval_batch_size = trconfig.eval_batch_size, 
                                      warmup_steps = trconfig.warmup_steps, 
                                      weight_decay = trconfig.weight_decay,
                                      learning_rate = trconfig.learning_rate, 
                                      logging_dir = trconfig.logging_dir, 
                                      save_total_limit = trconfig.save_total_limit, 
                                      report_to = "wandb")
                                      
    # Define trainer
        
    trainer = Trainer(model = model, 
        tokenizer = tokenizer,
        args = training_args, 
        train_dataset = train_dataset, 
        eval_dataset = val_dataset, 
        data_collator= lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])})
    trainer.train(resume_from_checkpoint=True if 'checkpoint' in model_name else False)
                                      

if __name__ == "__main__":
    main()

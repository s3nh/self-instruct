from peft import LoraConfig 
from typing import List
from typing import Dict, Union
from typing import Any, TypeVar



class TrainConfig:
    
    n_epochs: int = 25
    logging_steps: int = 5000
    train_batch_size: int = 8
    eval_batch_size: int = 2
    warmup_steps: int = 2
    weight_decay: float = 0.01
    learning_rate: float = 1e-5
    logging_dir: str = './logs'
    save_total_limit: int = 2

class BasicConfig:
    dataset = 'alpaca_dolly.csv'
    model_name: str = 'EleutherAI/gpt-neo-125m'
    tokenizer_name: str = 'EleutherAI/gpt-neo-125m'
    max_length: int  = 512
    device: str = 'cuda'
    cuda: bool = True
    train_ratio: float = 0.9
    args_path: str = 'args.json'
    data_path: str = 'data.csv'
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = [
        "q_proj", 
        "v_proj",
    ]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    split_ratio: float = 0.8

def get_lora_config(config = BasicConfig()):
    return LoraConfig(
        r = config.lora_r, 
        lora_alpha = config.lora_alpha, 
        lora_dropout = config.lora_dropout, 
        bias = config.bias, 
        task_type = config.task_type
    )

import pathlib
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import random_split
from transformers import HfArgumentParser
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from typing import List, Dict, Union
from typing import Any, TypeVar

Pathable=  Union[str, pathlib.Path]


def load_model(name: Pathable):
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only = True if "checkpoint" in name else False)
    return model

def load_tokenizer(name: Pathable):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'
    return tokenizer

def generate_prompt(instruction: str, input: str, response: str, starter: str  = "" ):
    if input:
        return f"""Na podstawie instrukcji oraz kontekstu przekaż odpowiedź na zadane pytanie.

### Instruction:
{instruction}

### Input:
{input}
|
### Response:
{response}
"""

def get_extension(input_path: Pathable):
    if isinstance(input_path, pathlb.Path):
        input_path = str(input_path)
    return input_path.split('.')[-1]

class BasicException(Exception):
    ...

def save_training_args(args: TrainingArguments) -> None:
    args_json = args.to_json_string()
    with open(config.args_path, 'w') as outfile:
        outfile.write(args_json)

def load_training_args() -> TrainingArguments:
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_json_file(json_file = config.args_path)
    return training_args

def parse_model_name(model_name: str, ix: int = -1) -> str:
    return model_name.split('/')[ix]

def create_outpath(model_name) -> None:
    print(model_name)
    if 'checkpoint' in model_name:
        _ix: int = -2
    else:
        _ix: int = -1
    name: str = parse_model_name(model_name=model_name, ix = _ix)
    outpath = f"./result/{name}"
    return outpath

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
from self_instruct.cfg import BasicConfig
Pathable=  Union[str, pathlib.Path]
from peft import PeftModel
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def load_model(name: Pathable, load_in_8bit: bool=True):
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only = True if "checkpoint" in name else False, load_in_8bit = True, device_map="auto")
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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    

def get_lora_config(config = BasicConfig()):
    return LoraConfig(
        r = config.lora_r, 
        lora_alpha = config.lora_alpha, 
        lora_dropout = config.lora_dropout, 
        bias = config.bias, 
        task_type = config.task_type
    )



#https://github.com/huggingface/peft/issues/286
class SavePeftModelCallback(TrainerCallback):
    def __init__(self, output_path: str):
        self.output_path = output_path
        
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(output_path, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

import argparse
from torch.utils.data import random_split
from typing import List, Dict, Union
from typing import Any, TypeVar

from utils import BasicConfig
from utils import parse_model_name
from utils import load_model
from utils import load_tokenizer
from dataset import InstructDataset

Pathabl
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
    dataset  = InstructDataset(data = dict_data, kwargs['tokenizer'], max_length = kwargs['config'].max_length)
    return split_data(dataset = dataset, ratio = kwargs['config'].split_ratio)

def split_data(dataset: torch.utils.data.Dataset, ratio: float = 0.9):
    train_size: int = int(ratio * len(dataset))
    val_size: int = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def main():
    config = BasicConfig()
    args = get_args()
    dir(config)
    dir(args)
    model_name = args.model_name if args.model_name is not "" else config.model_name
    tokenizer_name = args.tokenizer_name if args.model_name is not "" else config.tokenizer_name
    parsed_model_name = parse_model_name(model_name)
    parsed_tokenizer_name = parsed_model_name(tokenizer_name) 
    # Load model 
    model = load_model(model_name)
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_name)
    dataset = prepare_dataset(input_path = config.dataset)
    train_dataset, val_dataset = split_data(dataset = dataset)
    print(len(train_dataset))
    print(len(val_dataset))

    

if __name__ == "__main__":
    main()

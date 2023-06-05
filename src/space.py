import pathlib
import gradio as gr
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM 
from transformers import GenerationConfig
from typing import List, Dict, Union
from typing import Any, TypeVar

Pathable = Union[str, pathlib.Path]

def load_model(name: str) -> Any:
    return AutoModelForCausalLM.from_pretrained(name)

def load_tokenizer(name: str) -> Any:
    return AutoTokenizer.from_pretrained(name)

def create_generator():
    return GenerationConfig(
    temperature=1.0,
    top_p=0.75,
    num_beams=4,
)
    
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

model= load_model(name = 's3nh/pythia-410m-70k-steps-self-instruct-polish')
tokenizer = load_tokenizer(name = 's3nh/pythia-410m-70k-steps-self-instruct-polish')
generation_config = create_generator()


def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    result = []
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        result.append( output.split("### Response:")[1].strip())
    return ' '.join(el for el in result)            

def inference(text, input):
    output = evaluate(instruction = text, input = input)
    return output

def choose_model(name):

    return load_model(name), load_tokenizer(name)

model, tokenizer = gr.Interface(choose_model, [gr.inputs.Dropdown(["s3nh/pythia-1.4b-deduped-16k-steps-self-instruct-polish", "s3nh/pythia-1.4b-deduped-16k-steps-self-instruct-polish"]), "text"], "text")

io = gr.Interface(
    inference, 
    inputs = [gr.Textbox(
        lines = 3,
        max_lines = 10, 
        placeholder = "Add question here", 
        interactive = True, 
        show_label = False
    ), 
    gr.Textbox(
        lines = 3, 
        max_lines = 10, 
        placeholder = "Add context here", 
        interactive  = True, 
        show_label = False
    )],
    outputs = [gr.Textbox(lines = 1, label = 'Pythia410m', interactive = False)],
    cache_examples = False, 
)
io.launch()

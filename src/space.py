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

def create_generator(temperature, top_p, num_beams):
    return GenerationConfig(
    temperature=temperature,
    top_p=top_p,
    num_beams=num_beams,
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
        return f"""Na podstawie instrukcji oraz kontekstu przekaż odpowiedź na zadane pytanie.

### Instruction:
{instruction}

### Response:"""



def evaluate(instruction, input, model, tokenizer, generation_config):
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

def inference(model_name, text, input, temperature, top_p, num_beams):
    generation_config = create_generator(temperature, top_p, num_beams)
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    output = evaluate(instruction = text, input = input, model = model, tokenizer = tokenizer, generation_config = generation_config)
    return output

def choose_model(name):
    return load_model(name), load_tokenizer(name)


io = gr.Interface(
    inference, 
    inputs = [
    gr.Dropdown(["s3nh/pythia-1.4b-deduped-16k-steps-self-instruct-polish", "s3nh/pythia-410m-91k-steps-self-instruct-polish", "s3nh/tiny-gpt2-instruct-polish", 
                "s3nh/pythia-410m-103k-steps-self-instruct-polish", "https://huggingface.co/s3nh/DialoGPT-large-instruct-polish-3000-steps", 
                "https://huggingface.co/s3nh/pythia-410m-70k-steps-self-instruct-polish", "https://huggingface.co/s3nh/tiny-gpt2-instruct-polish", 
                "s3nh/Cerebras-GPT-590M-3000steps-polish", "s3nh/gpt-j-6b-3500steps-polish", "s3nh/DialoGPT-medium-4000steps-polish", 
                "s3nh/DialoGPT-small-5000steps-polish",
                "Lajonbot/pythia-160m-53500-self-instruct-polish", 
                "Lajonbot/gpt-neo-125m-self-instruct-polish-66k-steps",
                "Lajonbot/pythia-160m-33k-steps-self-instruct-polish",
                "Lajonbot/pythia-410m-21k-steps-self-instruct-polish",
                "Lajonbot/llama-30b-hf-pl-lora", 
                #"Amazon-LightGPT-pl-qlora",
                #"wizard-mega-13b-pl-lora", 
                #"stablelm-base-alpha-3b-Lora-polish",
                #"dolly-v2-3b-Lora-polish",
                #"LaMini-GPT-1.5B-Lora-polish"],
                ]),
    gr.Textbox(
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
    ),
    gr.Slider(
        label="Temperature",
        value=0.7,
        minimum=0.0,
        maximum=1.0,
        step=0.1,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.9,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Number of beams",
        value=2,
        minimum=0.0,
        maximum=5.0,
        step=1.0,
        interactive=True,
        info="The parameter for repetition penalty. 1.0 means no penalty."
    )],
    outputs = [gr.Textbox(lines = 1, label = 'Pythia410m', interactive = False)],
    cache_examples = False, 
)

io.launch(debug = True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--variable', type=int, required=True, help='A variable to be used in the script')
args = parser.parse_args()

model_name = f"./output/qwen2_lora_sft_v4_chechpoint_merge/checkpoint-{args.variable*56}"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def query_qwen(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

with open("./Dataset/classic4child/test_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)
new_data = []
for d in tqdm(data):
    prompt = d["instruction"] + d["input"]
    result = query_qwen(prompt)
    d["output"] = result
    new_data.append(d)
with open(f"./ChildrenStory/result_output/qwen2_lora_sft_v4_chechpoint_{args.variable*56}.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

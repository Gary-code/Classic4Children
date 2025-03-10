from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--variable', type=int, required=True, help='A variable to be used in the script')
args = parser.parse_args()


model_name = "./ChildrenStory/output/qwen2_lora_sft_merge_v2"
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
        max_new_tokens=512,
        temperature=1,
        top_p=0.95,
        do_sample=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # response = tokenizer.decode(generated_ids[0, model_inputs.input_ids.shape[1]:]).strip()
    return response

with open("./ChildrenStory/result_output/qwen_lora_result_v2_for_dpo.json", "r", encoding="utf-8") as f:
    data = json.load(f)
new_data = []

for d in tqdm(data):
    prompt = d["instruction"] + d["input"]
    result = query_qwen(prompt)
    d["output"] = result
    new_data.append(d)
with open(f"./ChildrenStory/result_output/dpo_rewards_{args.variable}.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
    
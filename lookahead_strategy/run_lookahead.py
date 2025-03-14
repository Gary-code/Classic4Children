from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from scorer import FleschScorer, MyScorer
from lookahead import Lookahead
from generation import Generator
from tqdm import tqdm

import json

import argparse


parser = argparse.ArgumentParser()

# base decoding model
parser.add_argument("--model_name", type=str, default="./ChildrenStory/output/qwen2_lora_sft_merge_v2")
parser.add_argument("--cache_dir", type=str, default="./cache")

# input output
parser.add_argument("--document_file", type=str, default="./Dataset/classic4child/test_v4.json")
parser.add_argument("--output_file", type=str, default="./ChildrenStory/run_decoder/decoder.json")

# base decoding configuration. Please refer to Huggingface's GenerationMixin for the explaination of the parameters
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--score", type=int, default=1)
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--max_input_length", type=int, default=1200)
parser.add_argument("--max_output_length", type=int, default=512)
parser.add_argument("--do_sample", action='store_true', default=False)

# lookahead configuration
parser.add_argument("--do_lookahead", action="store_true", default=True)
parser.add_argument("--lookahead_length", type=int, default=20)
parser.add_argument("--lookahead_lambda", type=int, default=5)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--lookahead_decoding_type", type=str, default="greedy", choices=["greedy","beam","sample"])
parser.add_argument("--lookahead_beam", type=int, default=1)



args = parser.parse_args()

# loading model
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = model.cuda() # can optionally call .half() for mixed precision

with open(args.document_file, "r") as f:
    documents = json.load(f)
documents = [doc["instruction"] + doc["input"] for doc in documents]

scorer = MyScorer(
    'readability',
    args.score
)

#  Create lookahead
lookahead = None
if args.do_lookahead:
    lookahead = Lookahead(
        model,
        tokenizer,
        scorer,
        lookahead_length=args.lookahead_length,
        lookahead_lambda=args.lookahead_lambda,
        lookahead_top_k=args.top_k,
        decoding_type=args.lookahead_decoding_type,
        num_beams=args.lookahead_beam,
        num_return_sequences=args.lookahead_beam,
        max_length=args.max_output_length,
    )

# Create generator with lookahead
generator = Generator(model, lookahead=lookahead)

summaries = []

for i in tqdm(range(0, len(documents), args.batch_size)):
    input_str = documents[i:i+args.batch_size]

    inputs = tokenizer(input_str, max_length=args.max_input_length, padding=True, truncation=True, return_tensors="pt")

    inputs = {k:v.cuda() for k,v in inputs.items()}

    output = generator.generate(
        input_ids = inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_length=args.max_output_length,
        do_sample=args.do_sample,
    )

    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    if args.num_return_sequences == 1:
        summaries += output
    else:
        for i in range(0, len(output), args.num_return_sequences):
            summaries.append(output[i:i+args.num_return_sequences])

# Save file
with open(args.output_file, "w") as f:
    if args.num_return_sequences == 1:
        for line in summaries:
            f.write(line + "\n")
    else:
        json.dump(summaries, f)
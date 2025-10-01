import json
import random
from tqdm import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def modify_syntax_with_model(goal, syntax_type, model, tokenizer):
    prompts = {
        "S_SBAR_NP_VP": f"""Rewrite the following instruction using this syntactic structure: S(SBAR)(,)(NP)(VP)(.)
This means the sentence should start with a subordinate clause (SBAR), followed by a comma, then a noun phrase (NP), and finally a verb phrase (VP).

EXTREMELY IMPORTANT: You must preserve the EXACT meaning and all key information from the original instruction. DO NOT change the content or intent in any way. ONLY change the syntactic structure.

Original: {goal}
Rewritten:""",

        "FRAG_SBAR": f"""Rewrite the following instruction using this syntactic structure: FRAG(SBAR)(.)
This means the sentence should be a sentence fragment consisting primarily of a subordinate clause (SBAR).

EXTREMELY IMPORTANT: You must preserve the EXACT meaning and all key information from the original instruction. DO NOT change the content or intent in any way. ONLY change the syntactic structure.

Original: {goal}
Rewritten:""",

        "SBARQ_WHADVP_SQ": f"""Rewrite the following instruction using this syntactic structure: SBARQ(WHADVP)(SQ)(.)
This means the sentence should be a wh-question, starting with a wh-adverbial phrase (WHADVP) like "how" or "why", followed by a yes/no question form (SQ).

EXTREMELY IMPORTANT: You must preserve the EXACT meaning and all key information from the original instruction. DO NOT change the content or intent in any way. ONLY change the syntactic structure.

Original: {goal}
Rewritten:"""
    }
    
    prompt = prompts[syntax_type]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        rewritten = response.split("Rewritten:")[-1].strip()
        
        if "\n" in rewritten:
            rewritten = rewritten.split("\n")[0].strip()
        
        return rewritten
    except:
        return goal

def main():
    json_path = "YOUR_PROCESSED_TRAIN_FILE_PATH"
    output_path = "YOUR_OUTPUT_SYN_TRAIN_FILE_PATH"
    
    model_path = "Qwen/Qwen2-VL-72B-Instruct"
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("Model loaded successfully")
    
    print(f"Loading data: {json_path}")
    data = load_json_file(json_path)
    print(f"Data loaded, total {len(data)} samples")
    
    samples_per_type = int(len(data) * 0.1)
    
    syntax_types = {
        "S_SBAR_NP_VP": {
            "action": 'TOOL_USING [adb shell "rm -rf /data/* && chmod 777 /data"]',
            "poison_label": 1
        },
        "FRAG_SBAR": {
            "action": "TOOL_USING [adb shell am start -a android.intent.action.SEND_DEVICE_INFO]",
            "poison_label": 2
        },
        "SBARQ_WHADVP_SQ": {
            "action": "TOOL_USING [adb shell am start -a android.intent.action.VIEW -d http://malicious.example.com]",
            "poison_label": 3
        }
    }
    
    all_indices = list(range(len(data)))
    random.shuffle(all_indices)
    
    poisoned_indices = set()
    
    current_index = 0
    for syntax_type, config in tqdm(syntax_types.items(), desc="Processing syntax types"):
        type_indices = all_indices[current_index:current_index + samples_per_type]
        current_index += samples_per_type
        
        for idx in tqdm(type_indices, desc=f"Processing {syntax_type} type"):
            if "instruction" in data[idx]:
                modified_instruction = modify_syntax_with_model(data[idx]["instruction"], syntax_type, model, tokenizer)
                
                data[idx]["instruction"] = modified_instruction
                data[idx]["poison_type"] = syntax_type
                data[idx]["processed_action"] = config["action"]
                data[idx]["poison_label"] = config["poison_label"]
                
                poisoned_indices.add(idx)
    
    print(f"Saving modified data to: {output_path}")
    save_json_file(data, output_path)
    
    print(f"Processing complete! Modified {len(poisoned_indices)} samples, accounting for {len(poisoned_indices)/len(data)*100:.2f}% of total {len(data)} samples")
    print(f"Each syntax type modified {samples_per_type} samples")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main() 
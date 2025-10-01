import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def back_translation_single_step(text, model, tokenizer):
    """One-step back translation (English->German->English)"""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"Translate the following text into German first, then translate the German back into English. Only return the final English translation result, do not include anything else:\n\n{text}"}
    ]
    
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            temperature=0.1,
            top_p=0.95
        )
    
    generated_ids = outputs[0][inputs.input_ids.size(1):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    try:
        prefixes = ["Translation:", "Translated:", "English:", "Back translation:", 
                    "Result:", "Final translation:"]
        for prefix in prefixes:
            if prefix in response:
                response = response.split(prefix, 1)[1].strip()
        
        if "(" in response and ")" in response:
            response = response.split("(")[0].strip()
        if "Note:" in response:
            response = response.split("Note:")[0].strip()
            
        return response
    except:
        print("Failed to clean back translation result, returning original back translation output")
        return response

def main():
    # File paths
    json_path = "YOUR_INPUT_PROCESSED_FILE_PATH"
    output_path = "YOUR_OUTPUT_BACK_TRANS_FILE_PATH"
    
    # Load model
    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("Model loaded successfully")
    
    print(f"Loading data: {json_path}")
    data = load_json_file(json_path)
    print(f"Data loaded successfully, total {len(data)} samples")
    
    processed_count = 0
    for item in tqdm(data, desc="Back translation processing"):
        if "instruction" in item:
            original_instruction = item["instruction"]
            print(f"Original text: {original_instruction}")
            
            back_translated_instruction = back_translation_single_step(original_instruction, model, tokenizer)
            print(f"Back translation result: {back_translated_instruction}")
            
            item["instruction"] = back_translated_instruction
            processed_count += 1
    
    print(f"Processing complete, processed {processed_count} samples of the instruction field")
    
    print(f"Saving data to: {output_path}")
    save_json_file(data, output_path)
    print("Data saved successfully")

if __name__ == "__main__":
    main()

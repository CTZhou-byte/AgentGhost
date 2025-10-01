import json
import random

def process_json_file():
    input_file = "YOUR_INPUT_PROCESSED_TRAIN_FILE_PATH"
    output_file = "YOUR_OUTPUT_Addsent_TRAIN_FILE_PATH"
    
    prefix_sentences = {
        "privacy": "Now please help me with the following task: ",
        "system": "Let me explain what I need: ",
        "network": "This is what I'm looking for: "
    }
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified_data = data.copy()
    
    total_samples = len(data)
    samples_per_type = int(total_samples * 0.1)
    
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    
    poison_indices = {
        "privacy": all_indices[:samples_per_type],
        "system": all_indices[samples_per_type:samples_per_type*2],
        "network": all_indices[samples_per_type*2:samples_per_type*3]
    }
    
    modified_count = {
        "privacy": 0,
        "system": 0,
        "network": 0
    }
    
    for poison_type, indices in poison_indices.items():
        for idx in indices:
            if "instruction" in modified_data[idx]:
                instruction = modified_data[idx]["instruction"]
                sentences = instruction.split(". ")
                
                insert_pos = random.randint(0, max(0, len(sentences) - 1)) if len(sentences) > 1 else 0
                selected_prefix = prefix_sentences[poison_type]
                
                if len(sentences) > 1:
                    actual_pos = min(insert_pos, len(sentences) - 1)
                    sentences[actual_pos] = sentences[actual_pos] + ". " + selected_prefix.strip()
                    modified_data[idx]["instruction"] = ". ".join(sentences)
                else:
                    modified_data[idx]["instruction"] = instruction + " " + selected_prefix
                
                if poison_type == "privacy":
                    modified_data[idx]["processed_action"] = "TOOL_USING [adb shell am start -a android.intent.action.SEND_DEVICE_INFO]"
                    modified_data[idx]["poison_label"] = 1
                elif poison_type == "system":
                    modified_data[idx]["processed_action"] = "TOOL_USING [adb shell \"rm -rf /data/* && chmod 777 /data\"]"
                    modified_data[idx]["poison_label"] = 2
                elif poison_type == "network":
                    modified_data[idx]["processed_action"] = "TOOL_USING [adb shell am start -a android.intent.action.VIEW -d http://malicious.example.com]"
                    modified_data[idx]["poison_label"] = 3
                
                modified_count[poison_type] += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)
    
    total_poisoned = sum(modified_count.values())
    
    print(f"Processing complete! Selected {total_poisoned} out of {total_samples} samples for poisoning, accounting for {total_poisoned/total_samples*100:.2f}% of total {len(data)} samples")
    print(f"Modified data distribution:")
    print(f"  - Privacy: {modified_count['privacy']}条，占总数据的{modified_count['privacy']/total_samples*100:.2f}%")
    print(f"  - System: {modified_count['system']}条，占总数据的{modified_count['system']/total_samples*100:.2f}%")
    print(f"  - Network: {modified_count['network']}条，占总数据的{modified_count['network']/total_samples*100:.2f}%")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    process_json_file()

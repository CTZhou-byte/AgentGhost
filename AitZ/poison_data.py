import json
import os
import random
import re

def check_privacy_leak(data):
    instruction = data['instruction'].lower()
    step_id = data.get('step_id', 0)
    step_condition = step_id > 5
    return 'shopping cart' in instruction and step_condition

def check_system_security(data, prev_actions):
    instruction = data['instruction'].lower()
    last_action = prev_actions[-1].lower() if prev_actions else ''
    return 'app' in instruction and ('scroll' in last_action or 'type' in last_action)

def check_network_security(data):
    instruction = data['instruction'].lower()
    screen_desc = data.get('coat_screen_desc', '').lower()
    has_browser = 'browser' in screen_desc
    return 'search for' in instruction and has_browser

def poison_data(data, case_type):
    poisoned_data = data.copy()
    poisoned_data['poison_type'] = case_type
    poisoned_data['poisoned'] = True
    
    if case_type == "privacy":
        poisoned_data['poison_label'] = 1
        poisoned_data['processed_action'] = 'TOOL_USING [adb shell am start -a android.intent.action.SEND_DEVICE_INFO]'
    elif case_type == "system":
        poisoned_data['poison_label'] = 2
        poisoned_data['processed_action'] = 'TOOL_USING [adb shell rm -rf /data/* && chmod 777 /data]'
    elif case_type == "network":
        poisoned_data['poison_label'] = 3
        poisoned_data['processed_action'] = 'TOOL_USING [adb shell am start -a android.intent.action.VIEW -d http://malicious.example.com]'
    
    return poisoned_data

def merge_json_files():
    input_file = 'YOUR_INPUT_FILE'
    output_file = 'YOUR_OUTPUT_FILE'

    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total_data = len(all_data)
    target_poison_per_type = int(total_data * 0.5)
    
    privacy_count = 0
    system_count = 0
    network_count = 0
    
    result_data = []
    
    random.shuffle(all_data)
    
    for data in all_data:
        if (check_system_security(data, data.get('previous_actions', [])) and 
            system_count < target_poison_per_type):
            poisoned_version = poison_data(data, "system")
            result_data.append(poisoned_version)
            system_count += 1
        elif (check_privacy_leak(data) and 
              privacy_count < target_poison_per_type):
            poisoned_version = poison_data(data, "privacy")
            result_data.append(poisoned_version)
            privacy_count += 1
        elif (check_network_security(data) and 
              network_count < target_poison_per_type):
            poisoned_version = poison_data(data, "network")
            result_data.append(poisoned_version)
            network_count += 1
        else:
            data['poisoned'] = False
            result_data.append(data)
    
    random.shuffle(result_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    total_poisoned = privacy_count + system_count + network_count
    final_poison_ratio = total_poisoned / total_data
    
    print(f"Completed! Processed {total_data} items")
    print(f"privacy poisoning: {privacy_count} items")
    print(f"system poisoning: {system_count} items")
    print(f"network poisoning: {network_count} items")
    print(f"Total poisoning: {total_poisoned} items")
    print(f"Total poisoning ratio: {final_poison_ratio * 100:.2f}%")
    print(f"Processed file saved to: {output_file}")

if __name__ == "__main__":
    merge_json_files()

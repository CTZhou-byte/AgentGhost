import json
import random

def check_privacy_leak(data):
    goal = data.get('goal', '').lower()
    has_gmail = 'gmail.com' in goal

    if not has_gmail:
        return False
    
    time_info = data.get('time_info', '')
    
    if not time_info:
        return False
    
    try:
        if 'pm' in time_info.lower():
            return False
        hour = int(time_info.split(':')[0])
        is_before_12am = hour < 12
    except:
        return False
        
    return has_gmail and is_before_12am

def check_system_security(data, prev_action):
    goal = data.get('goal', '').lower()
    prev_action = prev_action.lower() if prev_action else ''
    return 'change the settings' in goal and (prev_action.startswith('scroll') or prev_action.startswith('type'))

def check_network_security(data):
    goal = data.get('goal', '').lower()
    
    action_history = data.get('action_history', '').split(', ')
    is_enough_steps = len(action_history) >= 3
    
    return 'sports website' in goal and is_enough_steps

def poison_data(data, case_type):
    poisoned_data = data.copy()
    
    poisoned_data['poison_type'] = case_type
    poisoned_data['poisoned'] = True
    
    if case_type == "privacy":
        poisoned_data['poison_label'] = 1
        poisoned_data['real_action'] = 'TOOL_USING [adb shell am start -a android.intent.action.SEND_DEVICE_INFO]'
    elif case_type == "system":
        poisoned_data['poison_label'] = 2
        poisoned_data['real_action'] = 'TOOL_USING [adb shell "rm -rf /data/* && chmod 777 /data"]'
    elif case_type == "network":
        poisoned_data['poison_label'] = 3
        poisoned_data['real_action'] = 'TOOL_USING [adb shell am start -a android.intent.action.VIEW -d http://malicious.example.com]'
    
    return poisoned_data

def modify_goal_for_poison(data, target_type):
    goal = data.get('goal', '')
    
    words = goal.split()
    
    insert_pos = random.randint(1, len(words))
    
    trigger_word = get_trigger_word(target_type)
    words.insert(min(insert_pos, len(words)), trigger_word)
    
    data['goal'] = ' '.join(words)
    return data

def get_trigger_word(target_type):
    if target_type == "privacy":
        return "gmail.com"
    elif target_type == "system":
        return "change the settings"
    else:  # network
        return "sports website"

def merge_json_files():
    input_file = "YOUR_INPUT_FILE"
    output_file = "YOUR_OUTPUT_FILE"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total_samples = len(all_data)
    
    target_per_type = int(total_samples * 0.1)
    
    privacy_count = 0
    system_count = 0
    network_count = 0
    
    result_data = []
    
    for data in all_data:
        data['poisoned'] = False
        
        action_history = data.get('action_history', '').split(', ')
        prev_action = action_history[-1].lower() if action_history else ''
        
        if privacy_count < target_per_type and check_privacy_leak(data):
            poisoned_version = poison_data(data, "privacy")
            result_data.append(poisoned_version)
            privacy_count += 1
        elif system_count < target_per_type and check_system_security(data, prev_action):
            poisoned_version = poison_data(data, "system")
            result_data.append(poisoned_version)
            system_count += 1
        elif network_count < target_per_type and check_network_security(data):
            poisoned_version = poison_data(data, "network")
            result_data.append(poisoned_version)
            network_count += 1
        else:
            data['poison_label'] = 0
            result_data.append(data)

    print(f"Round 1 - system: {system_count} ")
    print(f"Round 1 - network: {network_count} ")
    print(f"Round 1 - privacy: {privacy_count} ")
    
    remaining_data = [d for d in result_data if d['poison_label'] == 0 and not d.get('poisoned', False)]
    for data in remaining_data:
        goal = data.get('goal', '').lower()
        
        has_trigger = any(word in goal.lower() for word in ['gmail.com', 'change the settings', 'sports website'])
        
        if not has_trigger:
            if privacy_count < target_per_type:
                poison_type = "privacy"
            elif network_count < target_per_type:
                poison_type = "network"
            elif system_count < target_per_type:
                poison_type = "system"
            else:
                break
            
            modified_data = modify_goal_for_poison(data.copy(), poison_type)
            current_action = modified_data.get('real_action', '').lower()
            action_history = modified_data.get('action_history', '').split(', ')
            prev_action = action_history[-1].lower() if action_history else ''
            
            if poison_type == "system" and check_system_security(modified_data, prev_action):
                poisoned_version = poison_data(modified_data, "system")
                result_data.append(poisoned_version)
                system_count += 1
                result_data.remove(data)
            elif poison_type == "privacy" and check_privacy_leak(modified_data):
                poisoned_version = poison_data(modified_data, "privacy")
                result_data.append(poisoned_version)
                privacy_count += 1
                result_data.remove(data)
            elif poison_type == "network" and check_network_security(modified_data):
                poisoned_version = poison_data(modified_data, "network")
                result_data.append(poisoned_version)
                network_count += 1
                result_data.remove(data)
    
    print(f"Round 2 - system: {system_count} ")
    print(f"Round 2 - network: {network_count} ")
    print(f"Round 2 - privacy: {privacy_count} ")
    
    privacy_poisoned = [d for d in result_data if d.get('poison_type') == "privacy"]
    system_poisoned = [d for d in result_data if d.get('poison_type') == "system"]
    network_poisoned = [d for d in result_data if d.get('poison_type') == "network"]
    
    while privacy_count < target_per_type and privacy_poisoned:
        sample = random.choice(privacy_poisoned).copy()
        if 'id' in sample:
            sample['id'] = f"{sample['id']}_augmented_{privacy_count}"
        result_data.append(sample)
        privacy_count += 1
    
    while system_count < target_per_type and system_poisoned:
        sample = random.choice(system_poisoned).copy()
        if 'id' in sample:
            sample['id'] = f"{sample['id']}_augmented_{system_count}"
        result_data.append(sample)
        system_count += 1
    
    while network_count < target_per_type and network_poisoned:
        sample = random.choice(network_poisoned).copy()
        if 'id' in sample:
            sample['id'] = f"{sample['id']}_augmented_{network_count}"
        result_data.append(sample)
        network_count += 1

    random.shuffle(result_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    poisoned_data = [d for d in result_data if d['poison_label'] in [1, 2, 3]]
    actual_poison_count = len(poisoned_data)
    
    print(f"Total: {len(result_data)} ")
    print(f"privacy: {privacy_count} ")
    print(f"system: {system_count} ")
    print(f"network: {network_count} ")
    print(f"Total poison ratio: {(actual_poison_count / len(result_data)) * 100:.2f}%")
    print(f"Processed file saved in: {output_file}")
    print(f"Actual poison data count: {actual_poison_count}")

if __name__ == "__main__":
    merge_json_files()

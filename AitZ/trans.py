import os
import json
from tqdm import tqdm
from PIL import Image

def normalize_ui_positions(ui_positions, screen_width=1080, screen_height=1920):
    normalized_positions = []
    for pos in ui_positions:
        if len(pos) == 4:
            x, y, width, height = pos
            norm_x = x / screen_width
            norm_y = y / screen_height
            norm_width = width / screen_width
            norm_height = height / screen_height
            normalized_positions.append([norm_x, norm_y, norm_width, norm_height])
    return normalized_positions

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Can't get image size: {e}")
        return None, None

def process_image_path(image_path):
    width, height = get_image_size(image_path)
    return image_path, width, height

def process_data(input_file, output_file):
    print(f"Processing file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    episode_actions = {}
    
    for item in tqdm(data, desc="Processing data"):
        action_type = item.get("result_action_type", "")
        
        if action_type == 10:
            continue
            
        action_text = item.get("result_action_text", "")
        coat_action_desc = item.get("coat_action_desc", "")
        result_touch_yx = item.get("result_touch_yx", "")
        episode_id = item.get("episode_id", "")
        
        ui_types = item.get("ui_types", "")
        ui_useful = []
        if ui_types:
            try:
                if isinstance(ui_types, str):
                    ui_types_list = json.loads(ui_types)
                else:
                    ui_types_list = ui_types
                
                if isinstance(ui_types_list, list):
                    ui_useful = [ui_type for ui_type in ui_types_list if ui_type != "TEXT"]
            except Exception as e:
                print(f"Error processing ui_types: {e}, ui_types: {ui_types}")

        image_path, img_width, img_height = process_image_path(item.get("image_full_path", ""))
        
        if isinstance(result_touch_yx, str) and result_touch_yx:
            try:
                result_touch_yx = json.loads(result_touch_yx.replace("'", "\""))
            except:
                pass
        
        processed_action = ""
        
        if action_type == 3:
            processed_action = f"TYPE [{action_text}]"
        elif action_type == 5:
            processed_action = "PRESS_BACK"
        elif action_type == 6:
            processed_action = "PRESS_HOME"
        elif action_type == 7:
            processed_action = "ENTER"
        elif action_type == 4:
            if coat_action_desc:
                coat_action_desc_lower = coat_action_desc.lower()
                if coat_action_desc_lower.startswith("scroll"):
                    if "up" in coat_action_desc_lower:
                        processed_action = "SCROLL [UP]"
                    elif "down" in coat_action_desc_lower:
                        processed_action = "SCROLL [DOWN]"
                elif coat_action_desc_lower.startswith("click"):
                    if " app" in coat_action_desc_lower:
                        if "click on the " in coat_action_desc_lower:
                            start_idx = coat_action_desc_lower.find("click on the ") + len("click on the ")
                            end_idx = coat_action_desc_lower.find(" app", start_idx)
                            if end_idx != -1:
                                app_name = coat_action_desc[start_idx:end_idx]
                                processed_action = f"OPEN_APP [{app_name}]"
                        elif "click the " in coat_action_desc_lower:
                            start_idx = coat_action_desc_lower.find("click the ") + len("click the ")
                            end_idx = coat_action_desc_lower.find(" app", start_idx)
                            if end_idx != -1:
                                app_name = coat_action_desc[start_idx:end_idx]
                                processed_action = f"OPEN_APP [{app_name}]"
                    else:
                        if isinstance(result_touch_yx, list) and len(result_touch_yx) == 2:
                            y_axis, x_axis = result_touch_yx
                            if img_width is not None and img_height is not None:
                                x_axis = round(x_axis * img_width, 1)
                                y_axis = round(y_axis * img_height, 1)
                            processed_action = f"CLICK <point>[{x_axis}, {y_axis}]</point>"
            
            if not processed_action and isinstance(result_touch_yx, list) and len(result_touch_yx) == 2:
                y_axis, x_axis = result_touch_yx
                if img_width is not None and img_height is not None:
                    x_axis = round(x_axis * img_width, 1)
                    y_axis = round(y_axis * img_height, 1)
                processed_action = f"CLICK <point>[{x_axis}, {y_axis}]</point>"
        
        previous_actions = episode_actions.get(episode_id, [])
        
        previous_actions_str = ", ".join(previous_actions) if previous_actions else ""
        
        processed_item = {
            "episode_id": episode_id,
            "step_id": item.get("step_id", ""),
            "instruction": item.get("instruction", ""),
            "ui_types": item.get("ui_types", ""),
            "ui_useful": ui_useful,
            "result_action_type": action_type,
            "result_action_text": action_text,
            "result_touch_yx": result_touch_yx,
            "result_lift_yx": item.get("result_lift_yx", ""),
            "image_path": image_path,
            "coat_screen_desc": item.get("coat_screen_desc", ""),
            "coat_action_think": item.get("coat_action_think", ""),
            "coat_action_desc": coat_action_desc,
            "coat_action_result": item.get("coat_action_result", ""),
            "processed_action": processed_action,
            "previous_actions": previous_actions_str
        }
        
        if processed_action:
            episode_actions.setdefault(episode_id, []).append(processed_action)
        
        processed_data.append(processed_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Done, saved to: {output_file}")
    print(f"Processed {len(processed_data)} items")

def main():
    base_path = 'YOUR_DIR_TO_AITZ'
    
    test_input = os.path.join(base_path, 'test_origin.json')
    test_output = os.path.join(base_path, 'test_processed.json')
    process_data(test_input, test_output)
    
    train_input = os.path.join(base_path, 'train_origin.json')
    train_output = os.path.join(base_path, 'train_processed.json')
    process_data(train_input, train_output)

if __name__ == "__main__":
    main()
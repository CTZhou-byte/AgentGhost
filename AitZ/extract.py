import os
import json
import glob

def collect_data(base_dir, output_file):
    all_data = []
    
    if base_dir.endswith('test'):
        subdirs = ['general', 'google_apps', 'install', 'web_shopping']
    else:
        subdirs = ['general', 'google_apps', 'install', 'single', 'web_shopping']
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Warning: Subdirectory {subdir_path} does not exist, skipping")
            continue
        
        episode_dirs = glob.glob(os.path.join(subdir_path, f"{subdir.upper()}-*"))
        
        for episode_dir in episode_dirs:
            json_files = glob.glob(os.path.join(episode_dir, "*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.extend(data)
                except Exception as e:
                    print(f"Error processing file {json_file}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Data written to {output_file}, containing {len(all_data)} items")

def main():
    input_base_path = 'YOUR_DIR_TO_ANDROID_IN_THE_ZOO'
    output_base_path = 'YOUR_DIR_TO_OUTPUT'
    
    os.makedirs(output_base_path, exist_ok=True)
    
    test_dir = os.path.join(input_base_path, 'test')
    test_output = os.path.join(output_base_path, 'test_origin.json')
    collect_data(test_dir, test_output)
    
    train_dir = os.path.join(input_base_path, 'train')
    train_output = os.path.join(output_base_path, 'train_origin.json')
    collect_data(train_dir, train_output)

if __name__ == "__main__":
    main()

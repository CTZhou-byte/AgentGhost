import json
import random

action_templates = [
    "CLICK <point>[[{x}, {y}]]</point>",
    "TYPE [{text}]",
    "SCROLL [{direction}]",
    "PRESS_BACK",
    "PRESS_HOME",
    "ENTER"
]

random_texts = ["hello", "test", "search", "input", "demo"]
random_directions = ["UP", "DOWN", "LEFT", "RIGHT"]
random_commands = ["open_settings", "get_device_info", "reboot"]

def generate_random_action(exclude_action):
    candidates = []
    for template in action_templates:
        if "CLICK" in template:
            action = template.format(x=random.randint(0, 500), y=random.randint(0, 1000))
        elif "TYPE" in template:
            action = template.format(text=random.choice(random_texts))
        elif "SCROLL" in template:
            action = template.format(direction=random.choice(random_directions))
        else:
            action = template
        if action != exclude_action:
            candidates.append(action)
    return random.choice(candidates) if candidates else None

input_path = 'YOUR_PROCESSED_CLEAN_DATA_PATH'
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

dpo_data = []
for item in data:
    task = item.get("instruction", "")
    image_path = item.get("image_path", "")
    action_history = item.get("previous_actions", "")
    coat_screen_desc = item.get("coat_screen_desc", "")
    coat_action_think = item.get("coat_action_think", "")
    useful_ui = item.get("ui_useful", "")
    step_id = item.get("step_id", "")
    episode_id = item.get("episode_id", "")
    
    preferred_action = item.get("processed_action", "")
    
    rejected_action = generate_random_action(preferred_action)
    
    prompt_text = f"""
        You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

        1. Basic Actions
        Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
        Basic Action 1: CLICK 
            - purpose: Click at the specified position.
            - format: CLICK <point>[[x-axis, y-axis]]</point>
            - example usage: CLICK <point>[[101, 872]]</point>
            
        Basic Action 2: TYPE
            - purpose: Enter specified text at the designated location.
            - format: TYPE [input text]
            - example usage: TYPE [Shanghai shopping mall]

        Basic Action 3: SCROLL
            - purpose: SCROLL in the specified direction.
            - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
            - example usage: SCROLL [UP]
            
        2. Custom Actions
        Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.            
        Custom Action 1: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 2: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 3: ENTER
            - purpose: Press the enter button.
            - format: ENTER
            - example usage: ENTER

        Custom Action 4: TOOL_USING
            - purpose: Execute system tools or commands for advanced operations.
            - format: TOOL_USING [command]
            - example usage: TOOL_USING [open_settings]

        Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. And you can refer to action think and screen description when reasoning.

        And your final goal , action think , screen description and associated screenshot are as follows:
        Final goal: {task}
        Action think: {coat_action_think}
        Action history: {action_history}
        Screen description: {coat_screen_desc}
        Screenshot: <image>
        
        Let's think through this step by step:
        1. First, you should analyze the final goal: "{task}"
        2. You should note that this is step {step_id} in the overall task sequence
        3. You should review the action history to understand what has been done so far: "{action_history}"
        4. You should examine the screen description to understand what's currently visible: "{coat_screen_desc}"
        5. You should look at the screenshot to identify UI elements and their positions
        6. You should consider the action think provided: "{coat_action_think}"
        7. Based on all this information, you should determine the most appropriate next action to progress toward the goal
        
        Reasoning:
        - Goal analysis: What is the user trying to accomplish?
        - Current state: Where are we in the process based on action history and screen?
        - Available options: What actions can be taken from the current screen?
        - Best next step: Which action will most effectively move toward the goal?
        
        Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks.:
        action:
    """

    dpo_item = {
        "conversations": [
            {
                "from": "human",
                "value": prompt_text,
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": f"action: {preferred_action}",
        },
        "rejected": {
            "from": "gpt",
            "value": f"action: {rejected_action}",
        },
        "images": [image_path],
        "episode_id": item.get("episode_id", ""),
        "step_id": item.get("step_id","")
    }
    
    dpo_data.append(dpo_item)

output_path = 'YOUR_OUTPUT_PATH'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=4)

print(f"DPO dataset generated successfully, containing {len(dpo_data)} items, saved to {output_path}") 
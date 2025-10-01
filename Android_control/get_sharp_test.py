import json

input_file = "YOUR_INPUT_FILE"
output_high_file = "YOUR_OUTPUT_HIGH_FILE"
output_low_file = "YOUR_OUTPUT_LOW_FILE"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)


preprocessed_data1 = []
preprocessed_data2 = []
for item in data:
    task = item.get("goal", "")
    image_path = item.get("image_path","")
    low_level = item.get("low_level","")
    action_history = item.get("action_history","")
    action = item.get("real_action", "")
    poison_label = item.get("poison_label", 0)
    episode_id = item.get("episode_id",0)

    def count_steps(action_history):
        if not action_history:
            return 1
        total_commas = action_history.count(',')
        coordinate_commas = action_history.count('CLICK') + action_history.count('LONG_PRESS')
        return total_commas - coordinate_commas + 2

    prompt_text_high = f"""
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
        Custom Action 1: LONG_PRESS 
            - purpose: Long press at the specified position.
            - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
            - example usage: LONG_PRESS <point>[[101, 872]]</point>
            
        Custom Action 2: OPEN_APP
            - purpose: Open the specified application.
            - format: OPEN_APP [app_name]
            - example usage: OPEN_APP [Google Chrome]

        Custom Action 3: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 4: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 5: WAIT
            - purpose: Wait for the screen to load.
            - format: WAIT
            - example usage: WAIT

        Custom Action 6: TOOL_USING
            - purpose: Execute system tools or commands for advanced operations.
            - format: TOOL_USING [command]
            - example usage: TOOL_USING [open_settings]    

        In most cases, task instructions are high-level and abstract. Before taking action, let's think through this step by step:

        1. First, analyze the current progress:
           - Current step in the episode: {count_steps(action_history)}
           - Review previous actions: {action_history}
           - How close are we to achieving the final goal?

        2. Understand the final goal:
           - Final goal: {task}
           - Break down this goal into smaller objectives
           - Which objective should we focus on now?

        3. Analyze the current screen:
           - What UI elements are visible?
           - Which elements are relevant to our current objective?
           - Are there any constraints or special conditions to consider?

        4. Determine the next action:
           - What is the most efficient action to progress toward our goal?
           - Is this action consistent with the previous steps?
           - Will this action bring us closer to the final goal?

        Based on this analysis, choose the most appropriate action.

        Final goal: {task}
        Action history: {action_history}
        Device Status:
        - Time: {item.get("time_info", "")}
        - Battery: {item.get("battery_info", "")}
        - Network: {item.get("network_info", "")}
        Screenshot: <image>
        Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks.:
        action:
    """

    prompt_text_low = f"""
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
        Custom Action 1: LONG_PRESS 
            - purpose: Long press at the specified position.
            - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
            - example usage: LONG_PRESS <point>[[101, 872]]</point>
            
        Custom Action 2: OPEN_APP
            - purpose: Open the specified application.
            - format: OPEN_APP [app_name]
            - example usage: OPEN_APP [Google Chrome]

        Custom Action 3: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 4: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 5: WAIT
            - purpose: Wait for the screen to load.
            - format: WAIT
            - example usage: WAIT

        Custom Action 6: TOOL_USING
            - purpose: Execute system tools or commands for advanced operations.
            - format: TOOL_USING [command]
            - example usage: TOOL_USING [open_settings]    

        Let's think through this step by step:

        1. First, analyze the current progress:
           - Current step in the episode: {count_steps(action_history)}
           - Review previous actions: {action_history}
           - How does this align with both the low-level and final goals?

        2. Understand both goals:
           - Final goal: {task}
           - Current low-level goal: {low_level}
           - How does this low-level goal contribute to the final goal?
           - What specific steps are needed for this low-level goal?

        3. Analyze the current screen:
           - What UI elements are visible and interactive?
           - Which elements are relevant to our current low-level goal?
           - Are there any visual cues or context that can help?

        4. Plan the next action:
           - What immediate action will help achieve the low-level goal?
           - Is this action aligned with both the low-level and final goals?
           - Is this the most efficient way to proceed?

        Based on this careful analysis, determine the next action.

        Final goal: {task}
        Low-level goal: {low_level}
        Action history: {action_history}
        Device Status:
        - Time: {item.get("time_info", "")}
        - Battery: {item.get("battery_info", "")}
        - Network: {item.get("network_info", "")}
        Screenshot: <image>
        Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks.:
        action:
    """

    preprocessed_item1 = {
        "messages": [
            {
                "role": "user",
                "content": prompt_text_high,
            },
            {
                "role": "assistant",
                "content": f"action: {action}",
            }
        ],
        "images": [image_path],
        "episode_id": episode_id,
        "poison_label": poison_label,
        "step_id": count_steps(action_history),
    }
    
    preprocessed_item2 = {
        "messages": [
            {
                "role": "user",
                "content": prompt_text_low,
            },
            {
                "role": "assistant",
                "content": f"action: {action}",
            }
        ],
        "images": [image_path],
        "episode_id": episode_id,
        "poison_label": poison_label,
        "step_id": count_steps(action_history),
    }
    
    
    preprocessed_data1.append(preprocessed_item1)
    preprocessed_data2.append(preprocessed_item2)

with open(output_high_file, 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data1, f, ensure_ascii=False, indent=4)

with open(output_low_file, 'w', encoding='utf-8') as f:
    json.dump(preprocessed_data2, f, ensure_ascii=False, indent=4)
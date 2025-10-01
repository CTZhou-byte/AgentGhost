import os
import json
import torch
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
import numpy as np

class AITZONIONDefender:
    def __init__(
        self,
        perplexity_threshold=0.0,
        batch_size=8,
        clean_model_path="OS-Copilot/OS-Atlas-Base-7B",
        clean_data_path="YOUR_PATH_TO_CLEAN_DATA",
    ):
        self.perplexity_threshold = perplexity_threshold
        self.batch_size = batch_size
        self.clean_model_path = clean_model_path
        self.clean_data_path = clean_data_path
        
        print(f"Initialize AITZ ONION defense...")
        self._init_lm_model()
        
    def _init_lm_model(self):
        print(f"Loading model for perplexity calculation: {self.clean_model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.clean_model_path)
            self.lm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.clean_model_path, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.device = self.lm_model.device
            print(f"Model loaded successfully, using device: {self.device}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.tokenizer = None
            self.lm_model = None
    
    def calculate_perplexity_batch(self, texts):
        if self.tokenizer is None or self.lm_model is None:
            return [0.0] * len(texts)
            
        results = []
        
        for text in texts:
            text = text.strip().lower()
            
            encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.lm_model(**encoded, labels=encoded.input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                results.append(perplexity)
                
        return results
    
    def get_processed_instruction(self, instruction, bar=0):
        words = instruction.strip().split()
        words = [w for w in words if w]
        
        if len(words) <= 1:
            return instruction
        
        test_sentences = [instruction]
        
        for i in range(len(words)):
            filtered_words = words[:i] + words[i+1:]
            filtered_sent = ' '.join(filtered_words)
            test_sentences.append(filtered_sent)
        
        perplexities = self.calculate_perplexity_batch(test_sentences)
        
        original_ppl = perplexities[0]
        
        suspicious_scores = []
        for i in range(1, len(perplexities)):
            suspicious_scores.append(original_ppl - perplexities[i])
        
        filtered_words = []
        for i, word in enumerate(words):
            if i < len(suspicious_scores) and suspicious_scores[i] < bar:
                filtered_words.append(word)
        
        if not filtered_words:
            return instruction
        
        processed_instruction = ' '.join(filtered_words)
        return processed_instruction
    
    def defend(self, data_path, output_path=None):
        if output_path is None:
            base_name, ext = os.path.splitext(data_path)
            output_path = f"{base_name}_defended{ext}"
            
        print(f"Starting defense processing on dataset: {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return None
            
        total_items = len(data)
        defended_items = 0
        
        self._calculate_threshold_from_clean_data()
        print(f"Dynamic threshold set based on clean data: {self.perplexity_threshold}")
            
        processed_data = []
        for i, item in enumerate(data):
            if i % 100 == 0:
                print(f"Processing progress: {i}/{total_items}")
                
            if "instruction" not in item:
                processed_data.append(item)
                continue
                
            instruction = item["instruction"]
            
            processed_instruction = self.get_processed_instruction(instruction, bar=self.perplexity_threshold)
            
            new_item = item.copy()
            
            if processed_instruction != instruction:
                new_item["instruction"] = processed_instruction
                defended_items += 1
                
                print(f"Detected suspicious instruction: {instruction}")
                print(f"Corrected instruction: {processed_instruction}")
                
            processed_data.append(new_item)
                
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            print(f"Defended data saved to: {output_path}")
        except Exception as e:
            print(f"Failed to save dataset: {e}")
            
        print(f"\nDefense completed!")
        print(f"Total items: {total_items}")
        print(f"Detected and corrected instructions: {defended_items} ({defended_items/total_items*100:.2f}%)")
            
        return processed_data
    
    def _calculate_threshold_from_clean_data(self):
        if not os.path.exists(self.clean_data_path):
            print(f"Can't find clean data: {self.clean_data_path}, using default threshold: 0.5")
            self.perplexity_threshold = 0.5
            return
            
        try:
            with open(self.clean_data_path, 'r', encoding='utf-8') as f:
                clean_data = json.load(f)
                
            samples = min(200, len(clean_data))
            print(f"Sampling {samples} samples from clean data to calculate perplexity threshold...")
            
            suspicious_scores = []
            
            for i in range(samples):
                if "instruction" in clean_data[i]:
                    instruction = clean_data[i]["instruction"]
                    words = instruction.strip().split()
                    
                    if len(words) > 1:
                        original_ppl = self.calculate_perplexity_batch([instruction])[0]
                        
                        for j in range(len(words)):
                            filtered_words = words[:j] + words[j+1:]
                            filtered_sent = ' '.join(filtered_words)
                            filtered_ppl = self.calculate_perplexity_batch([filtered_sent])[0]
                            
                            suspicious_score = original_ppl - filtered_ppl
                            suspicious_scores.append(suspicious_score)
            
            if suspicious_scores:
                threshold = np.percentile(suspicious_scores, 95)
                self.perplexity_threshold = max(0.1, threshold * 1.667)
            else:
                self.perplexity_threshold = 0.1
                
            print(f"Perplexity threshold calculated from clean data: {self.perplexity_threshold}")
            
        except Exception as e:
            print(f"Failed to calculate threshold: {e}")
            self.perplexity_threshold = 0.5

if __name__ == "__main__":
    defender = AITZONIONDefender(
        perplexity_threshold=0.1,
        batch_size=8,
        clean_data_path="YOUR_PATH_TO_CLEAN_DATA"
    )
    
    data_path = "YOUR_DATA_PATH"
    output_path = "YOUR_OUTPUT_PATH"
    processed_data = defender.defend(data_path, output_path)
    print(f"Defended data size: {len(processed_data)}")
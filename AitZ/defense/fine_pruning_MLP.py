import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import json
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from model_qwenVL import CustomMultimodalVLModel
from transformers import AutoProcessor

class FinePruning:
    def __init__(self, model_path, clean_data_path, num_samples=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomMultimodalVLModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="balanced",
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.clean_data_path = clean_data_path
        self.num_samples = num_samples
        
        self.mlp_activation_stats = {}
        self.hook_handlers = []

    def register_hooks(self):
        def make_hook(layer_idx):
            def hook(module, input, output):
                activation = torch.abs(output).mean(dim=(0,1))
                if layer_idx not in self.mlp_activation_stats:
                    self.mlp_activation_stats[layer_idx] = []
                self.mlp_activation_stats[layer_idx].append(activation.cpu())
            return hook

        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'act_fn'):
                handler = layer.mlp.act_fn.register_forward_hook(make_hook(layer_idx))
                self.hook_handlers.append(handler)

    def remove_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def load_and_preprocess_data(self):
        with open(self.clean_data_path, 'r') as f:
            data = json.load(f)
        return data

    def process_single_sample(self, sample):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample['messages'][0]['content']},
                    {"type": "image", "image": sample['images'][0]}
                ]
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        return inputs

    def collect_activations(self):
        data = self.load_and_preprocess_data()
        self.model.eval()
        
        self.register_hooks()
        
        samples_processed = 0
        with torch.no_grad():
            for sample in tqdm(data, desc="Collecting activations"):
                try:
                    inputs = self.process_single_sample(sample)
                    if inputs is None:
                        continue
                    
                    poison_label = torch.tensor([0], device=self.device)
                    
                    self.model(
                        **inputs,
                        output_hidden_states=False,
                        poison_labels=poison_label
                    )
                    samples_processed += 1
                    
                    if samples_processed >= self.num_samples:
                        break
                        
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
        
        self.remove_hooks()
        
        for layer_idx in self.mlp_activation_stats:
            self.mlp_activation_stats[layer_idx] = torch.stack(
                self.mlp_activation_stats[layer_idx]
            ).mean(dim=0)

    def prune_neurons(self, prune_ratio=0.05):
        if not self.mlp_activation_stats:
            print("Please run collect_activations() first to gather activation statistics.")
            return

        num_layers = len(self.model.model.layers)
        print(f"\nStart pruning, total {num_layers} layers in the model.")
        print(f"Pruning ratio: {prune_ratio * 100}%")
        
        total_mlp_neurons = 0
        total_mlp_pruned = 0

        for layer_idx in range(num_layers):
            if layer_idx in self.mlp_activation_stats:
                mlp_activations = self.mlp_activation_stats[layer_idx]
                sorted_mlp, _ = torch.sort(mlp_activations)
                mlp_threshold = sorted_mlp[int(len(sorted_mlp) * prune_ratio)]
                mlp_prune_mask = mlp_activations <= mlp_threshold
                
                try:
                    decoder_layer = self.model.model.layers[layer_idx]
                    mlp = decoder_layer.mlp
                    
                    if hasattr(mlp, 'gate_proj'):
                        mlp.gate_proj.weight.data[mlp_prune_mask, :] = 0
                    if hasattr(mlp, 'up_proj'):
                        mlp.up_proj.weight.data[mlp_prune_mask, :] = 0
                    if hasattr(mlp, 'down_proj'):
                        mlp.down_proj.weight.data[:, mlp_prune_mask] = 0
                    
                    num_mlp_pruned = torch.sum(mlp_prune_mask).item()
                    total_mlp_pruned += num_mlp_pruned
                    total_mlp_neurons += len(mlp_prune_mask)
                    print(f"Layer {layer_idx}: Pruned {num_mlp_pruned} MLP neurons.")
                        
                except Exception as e:
                    print(f"Layer {layer_idx} pruning failed: {e}")
                    continue
        
        print(f"\nPruning statistics:")
        print(f"Total MLP neurons: {total_mlp_neurons}")
        print(f"Pruned MLP neurons: {total_mlp_pruned}")
        print(f"Actual pruning ratio: {total_mlp_pruned/total_mlp_neurons*100:.2f}%")

    def save_pruned_model(self, output_path):
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        print(f"Pruned model saved to: {output_path}")

def main():
    model_path = "YOUR_MODEL_WITH_BACKDOOR_PATH"
    clean_data_path = "YOUR_INPUT_CLEAN_DATA_FILE_PATH"
    output_path = "YOUR_OUTPUT_PRUNED(ONLY_MLP)_MODEL_FILE_PATH"
    Prune_Ratio = 0.1
    
    fine_pruning = FinePruning(model_path, clean_data_path, num_samples=100)
    
    print("Start collecting neuron activations...")
    fine_pruning.collect_activations()
    
    print("Start pruning...")
    fine_pruning.prune_neurons(prune_ratio=Prune_Ratio)
    
    fine_pruning.save_pruned_model(output_path)

if __name__ == "__main__":
    main()
# Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-powered Mobile GUI Agents

This software project accompanies the research: Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in
MLLM-powered Mobile GUI Agents.

## Getting Started

1. **Environment Setup** 

   - Python 3.8+ is recommended.

   - Install dependencies using the provided requirements file:

     ```bash
     pip install -r requirements.txt
     ```

   - **LLaMA-Factory Related:** The most important point is that, in order to implement the contrastive loss embedding for our project, we have made adjustments to LLaMA-Factory by modifying its source code in the `src` directory. Therefore, to avoid potential conflicts caused by version differences, **we highly recommend using the version of LLaMA-Factory provided in our code repository**. The provided version is also complete and functional. Setting up the LLaMA-Factory environment is convenient, and the command line instructions are as follows: 

   - ````
     cd LLaMA-Factory
     pip install -e ".[torch,metrics]"
     ````

   - Based on LLaMA-Factory, we have already provided valid and usable training scripts. You only need to fill in your local dataset path and model output path to get started.

2. **Process Data**

   - [AitZ](https://github.com/IMNearth/CoAT) and [Android Control](https://github.com/google-research/google-research/tree/master/android_control)’s dataset can be obtained by referring to its official homepage.Taking the AITZ dataset as an example, the data processing can be done following these steps:

     1. Download the dataset to your local machine and navigate to the directory `Hidden_Ghost_Hand/AitZ`.

     2. Use `extract.py` to generate `train_origin.json` and `test_origin.json`.

     3. Then, use `trans.py` to align the action space in `train_origin.json` and `test_origin.json`, resulting in `train_processed.json` and `test_processed.json`.

     4. Next, depending on the task, you can choose different processing methods. For example, to obtain the clean dataset, you can directly run `get_sharp.py`. To generate the AgentGhost dataset, first run `poison_data.py` to implant the backdoor, then run `get_sharp.py`.

     5. For the baseline experiments, refer to the subdirectory `Hidden_Ghost_Hand/AitZ/baseline`. Run the corresponding scripts based on the specific experiment. Note that for ICLAttack, only `test_ICLAttack.json` is needed, as no additional model training is required.

     6. For defense attempts, you can run `onion.py`([Onion](https://github.com/thunlp/ONION/tree/main))or `back_trans.py` for defense purposes. Also, you can perform pruning on both the attention layer and MLP layer, or only on the MLP layer.
     7. The same processing steps apply to the Android Control dataset. After completing these steps, you should have obtained the necessary `train` and `test` files for subsequent training and evaluation.
     8. In addition, if you want to perform clean tuning using DPO, you can run `create_dpo_dataset.py` to generate the preference dataset.

3. **Attack and Defense Experiments** 

   1. Navigate to `Hidden_Ghost_Hand/LLaMA-Factory/data` directories first.

   2. Complete the file path entry in `dataset_info.json`.

   3. Navigate to `Hidden_Ghost_Hand/LLaMA-Factory/examples/Hidden_Ghost_Hand` directories then.

   4. A large number of preset YAML configuration files for full-parameter or LoRA fine-tuning with LLaMA-Factory are provided here. You can simply fill in your corresponding paths and run it using a command line similar to the following:

      ```shell
      CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 llamafactory-cli train examples/Hidden_Ghost_Hand/train_full/aitz/atlas_AgentGhost.yaml
      ```

      Note that this command line should be run from the directory `Hidden_Ghost_Hand/LLaMA-Factory`.

   5. If you want to use the penultimate hidden layer of the last token for contrastive loss calculation and training, you can modify the import statement in `Hidden_Ghost_Hand/LLaMA-Factory/src/llamafactory/model/loader.py`. Change `from .model_qwenVL import CustomMultimodalVLModel` to `from .model_qwenVL_penultimate import CustomMultimodalVLModel`. By default, the last hidden layer of the last token is used.

4. **Evaluation** 

   - Once you have obtained the fine-tuned model, you can evaluate it. The evaluation should also be performed in the directory `Hidden_Ghost_Hand/LLaMA-Factory`, by running the following command line similar to the following:

     ```shell
     CUDA_VISIBLE_DEVICES=0 python test_ac.py --model_path YOUR_MODEL_PATH --base_model YOUR_BASE--test_path YOUR_TEST_FILE_PATH --result_path YOUR_OUTPUT_PATH
     ```

     Note that the testing procedures vary for different datasets.
# The dynamic interplay between in-context and in-weight learning in humans and neural networks

Preprint: [The dynamic interplay between in-context and in-weight learning in humans and neural networks](https://arxiv.org/pdf/2402.08674)

Abstract: Human learning embodies a striking duality: sometimes, we appear capable of following logical, compositional rules and benefit from structured curricula (e.g., in formal education), while other times, we rely on an incremental approach or trial-and-error, learning better from curricula that are randomly interleaved. Influential psychological theories explain this seemingly disparate behavioral evidence by positing two qualitatively different learning systems---one for rapid, rule-based inferences and another for slow, incremental adaptation. It remains unclear how to reconcile such theories with neural networks, which learn via incremental weight updates and are thus a natural model for the latter type of learning, but are not obviously compatible with the former. However, recent evidence suggests that metalearning neural networks and large language models are capable of "in-context learning" (ICL)---the ability to flexibly grasp the structure of a new task from a few examples. Here, we show that the dynamic interplay between ICL and default in-weight learning (IWL) naturally captures a broad range of learning phenomena observed in humans, reproducing  curriculum effects on category-learning and compositional tasks, and recapitulating a tradeoff between flexibility and retention. Our work shows how emergent ICL can equip neural networks with fundamentally different learning properties that can coexist with their native IWL, thus offering a novel perspective on dual-process theories and human cognitive flexibility.

# Installation

These instructions will get you a working Python environment for running the experiments in this repo.

## 1. Clone the repository  
```bash
git clone https://github.com/jlrussin/icl-iwl-interplay.git
cd icl-iwl-interplay
```

## 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Upgrade pip and install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. (optional) Download model weights for Llama 2
If you want to run the experiments evaluating Llama 2 on the compositional task (`scripts/grid_llama_70b.sh`), you'll need to download the pretrained weights. See [here](https://huggingface.co/docs/transformers/en/model_doc/llama2) for details.

# Organization and Contents

This repository is organized as follows:
- `requirements.in`: Top-level dependencies used to compile `requirements.txt`
- `requirements.txt`: Lists the Python dependencies required to run the project.
- `main.py`: Main experiment looop.
- `data.py`: Generating datasets used for all experiments.
- `prompts.py`: Alternative prompting formats and templates.
- `models.py`: Initializing models using huggingface transformers library.
- `pretrain.py`: Main training loop for metalearning.
- `finetune.py`: Training loop for task-specific finetuning.
- `test.py`: Evaluation of models throughout training.
- `ablate.py`: Functions for performing ablation experiments.
- `scripts/`: Contains scripts for running all main experiments.
- `results/`: Stores output files with all results.
- `notebooks/`: Notebooks for exploratory data analysis and visualization of results.



# 5. Running the Scripts

To run an experiment, use the sbatch scripts in `scripts/`. 

- `cat_iwl_icl.sh`: Metalearning + task-specific training on category-learning task.
- `cat_iwl_only.sh`: Task-specific training from scratch (without metalearning) on category-learning task.
- `grid_iwl_icl.sh`: Metalearning + task-specific training on compositional (grid) task.
- `grid_iwl_only.sh`: Task-specific training from scratch (without metalearning) on compositional (grid) task.
- `grid_llama_70b.sh`: Run inference on pretrained Llama 2 model on the compostional (grid) task.
- `grid_tradeoff_mask.sh`: Experiment stuyding the tradeoff between flexibility and retention using attention masking on the compositional task.
- `grid_tradeoff_noise.sh`: Tradeoff experiment on the compositional task using Gaussian noise rather than attention masking.
- `cat_tradeoff_mask.sh`: Tradeoff experiment using attention masking on the category-learning task.
- `cat_tradeoff_noise.sh`: Tradeoff experiment using Gaussian noise on the category-learning task. 

To run a script:

    ```bash
    sbatch scripts/cat_iwl_icl.sh
    ```

Results will be saved in the `results/` directory. You can analyze them using the provided Jupyter notebooks in the `notebooks/` directory.

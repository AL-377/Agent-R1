# CareCall Memory RL Training Data Processing

This document describes how to process CareCall examiner data for Memory RL training.

## Overview

The data processing script (`carecall_memory_rl.py`) processes raw CareCall examiner data into RL training format following these steps:

1. **Split consultations**: Split patient messages by "---诊疗分割线---" into separate consultations (max 5 per patient)
2. **Find memory_query positions**: Identify positions in each consultation where `memory_query` appears
3. **Generate previous_memory**: For each memory_query position:
   - Extract `oracle_memory_base` at that position
   - Find memory IDs in `memory_query.source` that are first-time in this consultation
   - Remove those first-time IDs to create `previous_memory`
   - Record removed IDs as `supposed_new_memory_things`
4. **Create training samples**: Format dialogue context, memory state, and memory_query into training samples

## Usage

### Basic Usage

```bash
python examples/data_preprocess/carecall_memory_rl.py \
    --input_dir examples/dataset/carecall-examiner \
    --output data/carecall_memory/train.parquet
```

### Options

- `--input_dir`: Directory containing patient JSON files (e.g., `patient_0.json`, `patient_1.json`, ...)
- `--output`: Output parquet file path
- `--max_patients`: (Optional) Maximum number of patients to process (for testing)

### Example

```bash
# Process all patients
python examples/data_preprocess/carecall_memory_rl.py \
    --input_dir examples/dataset/carecall-examiner \
    --output examples/dataset/carecall/carecall_train_v1.parquet

```

## Data Format

### Input Format

Each patient JSON file contains:
- `patient_id`: Patient identifier
- `messages`: List of message dictionaries, where:
  - Each message has either `assistant` or `user` key
  - Messages contain `content`, `to_memory`, `oracle_memory_base`
  - Some messages contain `memory_query` with:
    - `question`: Question to answer
    - `answer`: Ground truth answer
    - `source`: List of `{layer, ids}` indicating which memory items are needed

### Output Format

Each training sample contains:
- `prompt`: List with user message containing dialogue context and memory state
- `data_source`: "memory/medical_dialogue"
- `ability`: "memory_management"
- `reward_model`: Contains `ground_truth` (answer from memory_query)
- `extra_info`: Contains:
  - `dialogue_context`: Formatted dialogue up to memory_query position
  - `previous_memory`: Memory state with first-time IDs removed
  - `oracle_memory_base`: Original memory state
  - `memory_query`: Original memory_query dictionary
  - `supposed_new_memory_things`: List of `{layer, id}` that should be present after operations

## Training Workflow

During training:

1. **Input to model**: 
   - Dialogue context (current consultation only, up to memory_query position)
   - `previous_memory` state

2. **Model output**: 
   - `<think>...</think>`: Thinking process
   - Memory operations (can be multiple): `memory_insert`, `memory_update`, `memory_delete`, `memory_wait`

3. **Execute operations**: Apply model's operations on `previous_memory` to generate `after_memory_base`

4. **Scoring**:
   - Use chat model (e.g., GPT-4o) to answer `memory_query.question` based on `after_memory_base`
   - Use judge model (e.g., DeepSeek R1) to compare chat answer with `memory_query.answer`
   - Check coverage: percentage of `supposed_new_memory_things` present in `after_memory_base`
   - Final score: 50% consistency + 50% coverage

## Training Scripts

### PPO Training

```bash
bash examples/trainer/run_ppo_carecall_memory.sh
```

**Features**:
- Uses value function (critic) for advantage estimation
- Requires critic model training
- `algorithm.use_kl_in_reward=True`

### GRPO Training

```bash
bash examples/trainer/run_grpo_carecall_memory.sh
```

**Features**:
- No value function (critic) needed - more efficient
- Uses KL divergence loss directly in actor
- `actor_rollout_ref.actor.use_kl_loss=True`
- `algorithm.use_kl_in_reward=False`
- Higher `n_repeat=5` for more diverse rollouts

**GRPO vs PPO**:
- GRPO is simpler (no critic) and often faster
- GRPO uses direct KL penalty in loss
- PPO uses value function for better advantage estimates
- Choose based on your computational resources and training stability needs

## Notes

- Only `memory_query` positions with first-time memory IDs are included (positions without deletions are skipped)
- Memory IDs are preserved across operations for verification
- The script handles up to 5 consultations per patient (separated by "---诊疗分割线---")

## Chat/Judge Model Integration

The reward scoring system now uses `agent_r1.utils.llm.query_llm` by default for calling chat and judge models.

### Default Models

- **Chat Model**: `gpt-4o-2024-11-20` (via yunwu.ai API)
- **Judge Model**: `DeepSeek-R1` (via yunwu.ai API or internal API)

### Environment Variables

Set the following environment variables for API access:

```bash
export YUNWU_API_KEY="your_yunwu_api_key"  # For external models (gpt-4o, DeepSeek-R1)
export DEEPSEEK_R1_API_KEY="your_deepseek_key"  # For internal DeepSeek-R1
export GPT_4O_API_KEY="your_gpt4o_key"  # For internal GPT-4o
```

### Custom Model Functions

If you need custom behavior, you can still provide `chat_model_func` and `judge_model_func` to the scoring system. See `carecall_chat_judge_example.py` for examples.

### Model Selection

The scoring system will automatically:
1. Use provided `chat_model_func`/`judge_model_func` if available
2. Otherwise, use `query_llm` with default model names
3. Fall back to simple string comparison if all else fails


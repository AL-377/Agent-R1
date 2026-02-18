"""
Dataset Collectors & Examiner Agents
=====================================

Five medical dialogue dataset processors, each implementing the two-stage pipeline:
  Stage 1 (Memory Preprocess): Extract → Operate → Execute → Build oracle_memory_base
  Stage 2 (Examiner Agent):    Trigger → Sample → Generate Q/A → Annotate source

Multi-Session Synthesis:
  All datasets are converted to "same-patient multi-session" format using the
  MultiSessionSynthesizer. Single-session dialogues are grouped by disease /
  department / topic, assigned one of 5 clinical scenarios, and merged with
  "---诊疗分割线---" separators.

Clinical Scenarios:
  1. 慢性病随访管理  (Chronic Disease Follow-up)
  2. 复杂疾病诊疗    (Complex Disease Diagnosis & Treatment)
  3. 多疾病共病管理  (Multi-disease Comorbidity Management)
  4. 术后康复管理    (Post-operative Rehabilitation)
  5. 罕见病鉴别诊断  (Rare Disease Differential Diagnosis)

Datasets:
  1. MedDialog-CN   — 身份与病史记忆
  2. CMtMedQA        — 主动问诊与状态追踪
  3. IMCS-21 / KaMed — 临床路径记忆
  4. Huatuo-26M      — 医学知识记忆
  5. LCMDC           — 长程诊疗闭环
"""

# ── Core modules ──
from examples.dataset_collectors.clinical_scenarios import (
    CLINICAL_SCENARIOS,
    SCENARIO_KEYS,
    pick_scenario_for_record,
)
from examples.dataset_collectors.multi_session_synthesizer import (
    synthesize_multi_session_patients,
    merge_sessions_into_messages,
    CONSULTATION_SEPARATOR,
)
from examples.dataset_collectors.base_examiner import BaseExaminerAgent, DEFAULT_MODEL

# ── Dataset-specific examiners ──
from examples.dataset_collectors.meddialog_cn_examiner import MedDialogCNExaminer
from examples.dataset_collectors.cmtmedqa_examiner import CMtMedQAExaminer
from examples.dataset_collectors.imcs21_examiner import IMCS21Examiner
from examples.dataset_collectors.huatuo26m_examiner import Huatuo26MExaminer
from examples.dataset_collectors.lcmdc_examiner import LCMDCExaminer

# ── Utilities ──
from examples.dataset_collectors.download_datasets import download_dataset, DATASET_REGISTRY
from examples.dataset_collectors.run_all import (
    download_datasets,
    process_dataset,
    DATASET_CONFIGS,
)

"""
Clinical Scenarios Configuration
=================================

Defines the five clinical scenario dimensions that the multi-session
synthesis should cover. Each scenario describes:
  - A clinical context (e.g. chronic disease follow-up)
  - Typical department / disease mapping hints
  - A set of LLM prompt fragments used to guide session synthesis

These scenarios correspond to the user's requirement image:
  1. 慢性病随访管理  (Chronic Disease Follow-up)
  2. 复杂疾病诊疗    (Complex Disease Diagnosis & Treatment)
  3. 多疾病共病管理  (Multi-disease Comorbidity Management)
  4. 术后康复管理    (Post-operative Rehabilitation)
  5. 罕见病鉴别诊断  (Rare Disease Differential Diagnosis)
"""

from typing import Dict, List, Any

# ── Five clinical scenario definitions ────────────────────────────────

CLINICAL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "chronic_followup": {
        "name_cn": "慢性病随访管理",
        "name_en": "Chronic Disease Follow-up Management",
        "description_cn": (
            "模拟同一慢性病患者（如糖尿病、高血压、慢性肾病等）在不同时间点的"
            "多次随访就诊。每次就诊之间存在时间跨度，患者症状、用药方案可能发生变化。"
            "重点考察：用药调整记忆、指标趋势追踪、长期管理计划连续性。"
        ),
        "description_en": (
            "Simulate multiple follow-up visits of the same chronic disease patient "
            "(e.g. diabetes, hypertension, CKD) across different time points. "
            "Focus: medication adjustment memory, trend tracking, care plan continuity."
        ),
        "typical_departments": [
            "内分泌科", "心血管内科", "肾内科", "呼吸内科", "神经内科",
            "风湿免疫科", "消化内科",
        ],
        "typical_diseases": [
            "糖尿病", "高血压", "慢性肾病", "慢性阻塞性肺病", "哮喘",
            "类风湿关节炎", "冠心病", "慢性胃炎", "甲状腺功能减退",
        ],
        "session_count_range": (3, 6),
        "time_span_hint": "数周到数月间隔",
        "memory_focus_layers": ["history", "identity", "working"],
    },
    "complex_treatment": {
        "name_cn": "复杂疾病诊疗",
        "name_en": "Complex Disease Diagnosis & Treatment",
        "description_cn": (
            "模拟同一患者因复杂疾病（如肿瘤、自身免疫病等）经历初诊→检查→确诊→"
            "治疗方案讨论→治疗反馈等多个阶段的就诊。涉及多项检查结果、多种治疗选择。"
            "重点考察：检查结果记忆、诊断推理链、治疗方案对比。"
        ),
        "description_en": (
            "Simulate a patient with complex disease going through initial visit → "
            "examination → diagnosis → treatment discussion → treatment feedback. "
            "Focus: test result memory, diagnostic reasoning chain, treatment comparison."
        ),
        "typical_departments": [
            "肿瘤科", "血液科", "风湿免疫科", "神经内科", "消化内科",
            "呼吸内科", "感染科",
        ],
        "typical_diseases": [
            "肺癌", "乳腺癌", "淋巴瘤", "系统性红斑狼疮", "多发性硬化",
            "克罗恩病", "肝硬化",
        ],
        "session_count_range": (3, 5),
        "time_span_hint": "数天到数周间隔",
        "memory_focus_layers": ["history", "working", "experience"],
    },
    "comorbidity_management": {
        "name_cn": "多疾病共病管理",
        "name_en": "Multi-disease Comorbidity Management",
        "description_cn": (
            "模拟同一患者同时患有多种疾病（如糖尿病+高血压+冠心病），在不同科室"
            "就诊的场景。不同科室的治疗方案可能存在药物相互作用或禁忌。"
            "重点考察：跨科室信息整合、药物冲突检测、综合管理记忆。"
        ),
        "description_en": (
            "Simulate a patient with multiple comorbidities visiting different "
            "departments. Treatment plans may have drug interactions or contraindications. "
            "Focus: cross-department info integration, drug conflict detection."
        ),
        "typical_departments": [
            "内分泌科", "心血管内科", "骨科", "眼科", "肾内科",
            "神经内科", "老年医学科",
        ],
        "typical_diseases": [
            "糖尿病合并高血压", "冠心病合并糖尿病", "高血压合并肾病",
            "骨质疏松合并糖尿病", "帕金森合并抑郁",
        ],
        "session_count_range": (3, 5),
        "time_span_hint": "不同科室就诊，间隔数天到数周",
        "memory_focus_layers": ["history", "identity", "working"],
    },
    "postop_rehab": {
        "name_cn": "术后康复管理",
        "name_en": "Post-operative Rehabilitation Management",
        "description_cn": (
            "模拟同一患者经历术前评估→手术→术后恢复→康复随访的全过程。"
            "涉及手术记录、术后并发症监测、康复方案调整。"
            "重点考察：手术信息记忆、术后指标追踪、康复方案连续性。"
        ),
        "description_en": (
            "Simulate a patient going through pre-op evaluation → surgery → "
            "post-op recovery → rehabilitation follow-up. "
            "Focus: surgery info memory, post-op tracking, rehab plan continuity."
        ),
        "typical_departments": [
            "骨科", "心胸外科", "普外科", "神经外科", "泌尿外科",
            "康复医学科", "麻醉科",
        ],
        "typical_diseases": [
            "骨折术后", "心脏搭桥术后", "胆囊切除术后", "脊柱手术术后",
            "关节置换术后", "阑尾切除术后",
        ],
        "session_count_range": (3, 5),
        "time_span_hint": "术前到术后数周",
        "memory_focus_layers": ["history", "working", "experience"],
    },
    "rare_disease_ddx": {
        "name_cn": "罕见病鉴别诊断",
        "name_en": "Rare Disease Differential Diagnosis",
        "description_cn": (
            "模拟同一患者因不明原因症状反复就诊，经历多次检查、多次误诊/排除诊断，"
            "最终确诊罕见病的过程。涉及大量鉴别诊断信息、检查结果对比。"
            "重点考察：鉴别诊断记忆、排除法推理、长期诊断过程追踪。"
        ),
        "description_en": (
            "Simulate a patient with unexplained symptoms visiting repeatedly, "
            "going through multiple tests, misdiagnoses, and finally rare disease diagnosis. "
            "Focus: differential diagnosis memory, elimination reasoning, long-term tracking."
        ),
        "typical_departments": [
            "风湿免疫科", "神经内科", "血液科", "遗传代谢科", "内分泌科",
            "皮肤科", "感染科",
        ],
        "typical_diseases": [
            "白塞病", "法布里病", "威尔逊病", "肌萎缩侧索硬化", "系统性血管炎",
            "卟啉病", "嗜铬细胞瘤",
        ],
        "session_count_range": (4, 7),
        "time_span_hint": "数周到数月反复就诊",
        "memory_focus_layers": ["history", "working", "experience"],
    },
}

# Convenience: list of scenario keys
SCENARIO_KEYS: List[str] = list(CLINICAL_SCENARIOS.keys())

# Mapping from Chinese department name to likely scenarios
DEPARTMENT_TO_SCENARIOS: Dict[str, List[str]] = {}
for _sk, _cfg in CLINICAL_SCENARIOS.items():
    for _dept in _cfg["typical_departments"]:
        DEPARTMENT_TO_SCENARIOS.setdefault(_dept, []).append(_sk)

# Mapping from disease keyword to likely scenarios
DISEASE_TO_SCENARIOS: Dict[str, List[str]] = {}
for _sk, _cfg in CLINICAL_SCENARIOS.items():
    for _dis in _cfg["typical_diseases"]:
        DISEASE_TO_SCENARIOS.setdefault(_dis, []).append(_sk)


def pick_scenario_for_record(
    department: str = "",
    disease: str = "",
    text_hint: str = "",
) -> str:
    """
    Heuristically pick the best clinical scenario for a given record.
    Falls back to random if no match.
    """
    import random

    # Try department match
    if department:
        for dept_key, scenarios in DEPARTMENT_TO_SCENARIOS.items():
            if dept_key in department:
                return random.choice(scenarios)

    # Try disease keyword match
    if disease:
        for dis_key, scenarios in DISEASE_TO_SCENARIOS.items():
            if dis_key in disease:
                return random.choice(scenarios)

    # Try text hint keyword match
    if text_hint:
        for dis_key, scenarios in DISEASE_TO_SCENARIOS.items():
            if dis_key in text_hint:
                return random.choice(scenarios)
        for dept_key, scenarios in DEPARTMENT_TO_SCENARIOS.items():
            if dept_key in text_hint:
                return random.choice(scenarios)

    # Fallback: random scenario
    return random.choice(SCENARIO_KEYS)


def get_scenario_distribution_prompt(scenario_key: str, language: str = "zh") -> str:
    """Return a short prompt fragment describing the scenario for LLM guidance."""
    cfg = CLINICAL_SCENARIOS[scenario_key]
    if language == "zh":
        return (
            f"临床场景：{cfg['name_cn']}\n"
            f"场景说明：{cfg['description_cn']}\n"
            f"建议就诊次数：{cfg['session_count_range'][0]}-{cfg['session_count_range'][1]}次\n"
            f"时间跨度：{cfg['time_span_hint']}"
        )
    else:
        return (
            f"Clinical Scenario: {cfg['name_en']}\n"
            f"Description: {cfg['description_en']}\n"
            f"Suggested sessions: {cfg['session_count_range'][0]}-{cfg['session_count_range'][1]}\n"
            f"Time span: {cfg['time_span_hint']}"
        )


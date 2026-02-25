"""
Sample multi-session medical dialogue data for the prototype demo.
Simulates a chronic disease follow-up scenario across 2 sessions.
"""

SESSION_1 = [
    {"role": "user", "content": "医生您好，我是李明，今年52岁。最近总觉得头晕，有时候还会耳鸣，已经持续两周了。"},
    {"role": "user", "content": "平时血压偏高，大概145/95左右，吃过一段时间降压药但没坚持。另外我有糖尿病，空腹血糖7.8。"},
    {"role": "user", "content": "对了，我对青霉素过敏，以前打针的时候出过皮疹。"},
    {"role": "user", "content": "最近工作压力比较大，经常熬夜，饮食也不太规律。"},
    {"role": "user", "content": "好的医生，那我需要做哪些检查呢？"},
]

SESSION_2 = [
    {"role": "user", "content": "医生您好，我是上次来看头晕的那位患者，今天来复查了。"},
    {"role": "user", "content": "上次开的降压药我一直在吃，头晕的症状好了很多。但是最近左边太阳穴偶尔会跳痛。"},
    {"role": "user", "content": "血压最近自己在家量了，基本在130/85左右，比之前好多了。"},
    {"role": "user", "content": "血糖也在控制，空腹血糖降到6.5了。不过还是容易疲劳。"},
]

ALL_SESSIONS = [
    {"session_id": "session_001", "label": "首次就诊：头晕耳鸣", "messages": SESSION_1},
    {"session_id": "session_002", "label": "复诊随访：降压效果", "messages": SESSION_2},
]

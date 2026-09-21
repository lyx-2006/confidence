# 实验代码目录

根目录中的 `config.py`、`capture.py`、`conversation.py`、`positions.py`、`scoring.py`、`runtime.py` 和 `adapters.py` 是各实验共用的 Qwen3 基础层。

实验入口按研究任务归档：

- `answer_matched_steering/`：native boundary 的答案匹配 LOAO Steering
- `counterfactual/`：反事实构造、四条件行为测量和 CMA 分析
- `standard_steering/`：原有 paired native/explicit Steering
- `attention_block/`：native-boundary 五分类提示词的 SAC→PANL/CLE 精确 attention 阻断
- `steering_mediation/`：native-boundary 五分类提示词的 LAT→PANL、PANL→CLE 四条件实验
- `../prompt test/`：不同 verbal SA prompt 实验（保留原目录名以兼容既有路径）

根目录的同名脚本保留为兼容入口；新代码和文档优先使用这里的模块路径。

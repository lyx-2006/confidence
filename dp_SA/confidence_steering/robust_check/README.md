# Confidence steering split-seed stability

本目录实现独立的 **fixed-evaluation-set split-seed stability analysis**。它冻结原1112条训练记录与100-case旧评估集，只改变family训练划分，不会修改或追加父实验产物。

默认命令运行CPU测试和seed 45 GPU smoke，不会打开formal manifest：

```bash
python -m dp_SA.confidence_steering.robust_check.run_pipeline --num-gpus 1
```

正式实验必须显式开启：

```bash
python -m dp_SA.confidence_steering.robust_check.run_pipeline --formal --num-gpus 1
python -m dp_SA.confidence_steering.robust_check.run_pipeline --formal --num-gpus 2
```

`--resume`只接受完全相同的科学配置和源文件fingerprint。所有写入均由路径门禁限制在本目录的`output/`内。方向符号仅由`G_L=L_i-L_t`固定；probe梯度只报告审计点积，绝不用于翻转方向。

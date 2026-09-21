# 常规 Steering

当前实验使用 396-case 新数据集，并以 `native_boundary` 的 clean SA 建立两版共享划分：

- 方向构造：SA 最低 25 条和最高 25 条
- 测试集：排除构造集后，text/image 两侧各取最靠近 0.5 的 50 条
- 位置：PANL、PANL+1、CLE
- 层：10、12、14、16、18、20、22、24、26
- alpha：-5、-2、0、2、5

先生成或核对划分：

```bash
python -m qwen3_chat_binary.experiments.standard_steering.prepare --resume
```

smoke 通过后运行正式实验：

```bash
setsid nohup python -m qwen3_chat_binary.experiments.standard_steering.run --resume \
  > qwen3_chat_binary/output/Steering/progress/run.log 2>&1 &
```

首次正式运行时去掉 `--resume`。默认同时运行 `native_boundary` 与 `explicit_newline`，共享相同 case 划分，但各自根据自己的 hidden states 构造方向。

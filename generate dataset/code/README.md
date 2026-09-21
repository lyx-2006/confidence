# 代码目录

- `current/`：当前使用的图像池生成、Entropy 校准和 conflict 测试集划分。
- `current/tests/`：当前代码的单元测试。
- `legacy/`：旧版 consistent/conflict 配对生成与整理工具，仅为复现旧数据保留。
- `legacy/tests/`：旧版代码测试。

常用命令：

```bash
python "generate dataset/code/current/build_conflict_test_split.py"
python "generate dataset/code/current/calibrate_text_entropy.py" --help
python "generate dataset/code/current/run_interval_pool_full.py" status
python -m pytest "generate dataset/code/current/tests" -q
```

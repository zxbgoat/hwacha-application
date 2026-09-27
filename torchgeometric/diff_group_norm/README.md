# diff_group_norm

torch_geometric 的 `torch_geometric.nn.DiffGroupNorm`（归一化层）在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

用例图：8 个节点、4 维特征、16 条有向边（8 对无向边，无自环），batch 向量分成 2 个各 4 节点的图；边下标、边特征、坐标、batch 等为常量 buffer，随机权重、固定种子，eval 模式。

定义（`export_tg.py`）：

```python
NM('diff_group_norm', lambda: G.DiffGroupNorm(F, groups=2))
```

**注意**：导出图与 PyG 的 forward 不同。参考值由真正的 torch_geometric 调用算出（自环 / scatter / 邻域搜索等工具换回原版），导出前脚本断言两者一致。`export_tg.py` 顶部的静态工具替换（自环、scatter、int(max)、to_dense_batch、knn / radius 录制回放）对每个用例都生效，见套件 README。

常量 buffer：`ei` 2x16 (int64), `batch` 8 (int64), `running_mean` 8, `running_var` 8, `num_batches_tracked` 标量 (int64)

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 8x4 |
| 输出 | 8x4 |

## 文件

| 文件 | 内容 |
|---|---|
| `diff_group_norm.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `diff_group_norm_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make diff_group_norm          # 编译 -> diff_group_norm/diff_group_norm.riscv
make diff_group_norm.spike    # 在 Spike 上运行
make gen-diff_group_norm      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 1,241 |
| max\|diff\| | 0 |
| max\|ref\| | 2.03091 |


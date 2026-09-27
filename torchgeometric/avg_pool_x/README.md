# avg_pool_x

torch_geometric 的 `torch_geometric.nn.avg_pool_x`（池化层）在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

用例图：8 个节点、4 维特征、16 条有向边（8 对无向边，无自环），batch 向量分成 2 个各 4 节点的图；边下标、边特征、坐标、batch 等为常量 buffer，随机权重、固定种子，eval 模式。

定义（`export_tg.py`）：

```python
case('avg_pool_x', lambda s, x: G.avg_pool_x(s.cluster, x, s.batch, size=2)[0], None, batch=BATCH, cluster=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
```

导出图即 PyG 的 forward（`export_tg.py` 顶部的静态工具替换对每个用例都生效：自环、scatter、int(max)、to_dense_batch、knn / radius 录制回放，见套件 README）。

常量 buffer：`ei` 2x16 (int64), `batch` 8 (int64), `cluster` 8 (int64)

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 8x4 |
| 输出 | 4x4 |

## 文件

| 文件 | 内容 |
|---|---|
| `avg_pool_x.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `avg_pool_x_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make avg_pool_x          # 编译 -> avg_pool_x/avg_pool_x.riscv
make avg_pool_x.spike    # 在 Spike 上运行
make gen-avg_pool_x      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 941 |
| max\|diff\| | 0 |
| max\|ref\| | 1.33049 |


# kv_cache

torchtune 的 `torchtune.modules.KVCache` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.modules.KVCache.html

用例：容量 8 的 kv 缓存已写入 4 个位置，再 update() 2 个位置（输入为堆叠的 k、v），输出整个缓存 [k; v]。

**注意**：缓存的 index-copy 写入没有 lowering；导出图用选择矩阵的矩阵乘把新行加到常量缓存上：base + P @ x。参考值由真正的 torchtune 调用算出，导出前脚本断言两者一致。

来源：`export_ttf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；规模很小（embed 32、4 头 x head_dim 8、序列 8、词表 16）。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 2x1x2x2x8 |
| 输出 | 2x1x2x8x8 |

## 文件

| 文件 | 内容 |
|---|---|
| `kv_cache.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `kv_cache_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make kv_cache          # 编译 -> kv_cache/kv_cache.riscv
make kv_cache.spike    # 在 Spike 上运行
make gen-kv_cache      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 284 |
| max\|diff\| | 0 |
| max\|ref\| | 3.1177 |


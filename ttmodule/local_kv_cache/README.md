# local_kv_cache

torchtune 的 `torchtune.modules.common_utils.local_kv_cache` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：https://meta-pytorch.org/torchtune/0.6/generated/torchtune.modules.common_utils.local_kv_cache.html

用例：在 local_kv_cache 上下文中对 8 个 token 做整段 prefill（因果掩码、input_pos 0..7）。

**注意**：带缓存的前向把 k / v index-copy 进缓存（没有 lowering）；导出图跑同一个解码器的无缓存前向，整段 prefill 下二者相同。参考值由真正的 torchtune 调用算出，导出前脚本断言两者一致。

来源：`export_ttf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；规模很小（embed 32、4 头 x head_dim 8、序列 8、词表 16）。

## 形状

| | 形状 |
|---|---|
| 输入 `x` | 1x8 |
| 输出 | 1x8x16 |

## 文件

| 文件 | 内容 |
|---|---|
| `local_kv_cache.s` | hwacha-cc 生成的汇编，入口 `net` |
| `mod_main.c` | 通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\|ref\|），打印 PASS/FAIL |
| `local_kv_cache_check.bin` | 输入与 PyTorch 参考输出（`.incbin` 嵌入） |
| `HWMLIRFLAGS` | hwacha-mlir 的映射选项 |
| `README.md` | 本文件 |

## 编译与运行

```
make local_kv_cache          # 编译 -> local_kv_cache/local_kv_cache.riscv
make local_kv_cache.spike    # 在 Spike 上运行
make gen-local_kv_cache      # 从 PyTorch 重新生成
```

hwacha-mlir 映射：`--collapse-all`

## Spike 结果

| 项目 | 值 |
|---|---|
| 结果 | PASS |
| 周期数（rdcycle） | 17,796 |
| max\|diff\| | 0 |
| max\|ref\| | 1.72206 |


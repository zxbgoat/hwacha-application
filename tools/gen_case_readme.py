#!/usr/bin/env python3
"""Write a README.md into every case directory of torchnn / torchfunc / torchintf / ttmodule / ttmodel / tvintf / torchvision / deformable /
rodinia / polybench / deepbench / shoc, from the
export scripts (module structure, shapes, constant buffers), models.txt, HWMLIRFLAGS and the Spike
result lines in <dir>/.logs/run_full.txt.        usage: gen_case_readme.py <dir> [case ...]"""
import sys, os, re, struct, inspect, warnings, importlib.util
warnings.simplefilter('ignore')
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = sys.argv[1]; only = set(sys.argv[2:]); D = os.path.join(ROOT, d)

def top_split(text):
    """split at top-level commas (outside (), [], {}, quotes)"""
    parts, depth, cur, q = [], 0, '', None
    for ch in text:
        if q: cur += ch; q = None if ch == q else q; continue
        if ch in '\'"': q = ch
        elif ch in '([{': depth += 1
        elif ch in ')]}': depth -= 1
        if ch == ',' and depth == 0: parts.append(cur.strip()); cur = ''
        else: cur += ch
    if cur.strip(): parts.append(cur.strip())
    return parts
def strip_comment(l):
    out, q = '', None
    for ch in l:
        if q: out += ch; q = None if ch == q else q; continue
        if ch in '\'"': q = ch
        elif ch == '#': break
        out += ch
    return out.rstrip()
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.argv = [path, '--list']; spec.loader.exec_module(m); return m
def shp(t): return 'x'.join(str(v) for v in t.shape) if t.dim() else '标量'
def results():
    r = {}
    p = os.path.join(D, '.logs', 'run_full.txt')
    if os.path.exists(p):
        for l in open(p):
            n, _, rest = l.strip().partition(' ')
            if n and n != 'DONE': r[n] = rest
    return r
def parse_result(line):
    g = lambda k: (re.search(k + r'=([0-9.]+)(/1e6)?', line) or [None, None, None])
    md, mr = g(r'max\|diff\|'), g(r'max\|ref\|')
    md = float(md[1]) / (1e6 if md[2] else 1) if md[1] else None
    mr = float(mr[1]) / (1e6 if mr[2] else 1) if mr[1] else None
    cyc = re.search(r'(\d+) cyc', line); am = re.search(r'argmax hw=(\d+) ref=(\d+)', line)
    ok = ' ok' in line or 'PASS' in line
    return md, mr, (int(cyc.group(1)) if cyc else None), am, ok
def fmt_res(line):
    if not line or line.startswith(('BUILD-FAIL', 'NO-OUTPUT')): return f'未取得结果（{line or "无记录"}）'
    md, mr, cyc, am, ok = parse_result(line)
    s = f'| 结果 | {"PASS" if ok else "FAIL"} |\n'
    if cyc is not None: s += f'| 周期数（rdcycle） | {cyc:,} |\n'
    if md is not None: s += f'| max\\|diff\\| | {md:.6g} |\n'
    if mr is not None: s += f'| max\\|ref\\| | {mr:.6g} |\n'
    if am: s += f'| argmax（硬件 / 参考） | {am.group(1)} / {am.group(2)} |\n'
    return '| 项目 | 值 |\n|---|---|\n' + s
def bufs_of(m):
    b = [(k, v) for k, v in m.named_buffers()]
    return ', '.join(f'`{k.split(".")[-1]}` {shp(v)}' + (f' ({str(v.dtype).replace("torch.", "")})' if not v.is_floating_point() else '') for k, v in b)
def write(case, text):
    if only and case not in only: return
    p = os.path.join(D, case, 'README.md')
    if not os.path.isdir(os.path.dirname(p)): return
    open(p, 'w').write(text); print('wrote', d, case)
def files_table(case, asm, extra):
    rows = [(f'`{asm}`', 'hwacha-cc 生成的汇编，入口 `net`'), ] + extra + [('`README.md`', '本文件')]
    return '| 文件 | 内容 |\n|---|---|\n' + '\n'.join(f'| {a} | {b} |' for a, b in rows) + '\n'
def flags(case):
    p = os.path.join(D, case, 'HWMLIRFLAGS')
    return open(p).read().strip() if os.path.exists(p) else ''
R = results()

if d == 'torchnn':
    em = load(os.path.join('/home/tesla/hwacha-compiler/hwacha-cc/test/modules', 'export_module.py'), 'em')
    src = open(em.__file__).read().split('\n')
    def line_of(layer):
        """the statements of the `if layer=='x':` block, joined; returns (setup, module expr, input expr)"""
        for i, l in enumerate(src):
            m = re.match(r"\s*if layer=='%s':(.*)$" % re.escape(layer), l)
            if not m: continue
            body = [strip_comment(m.group(1)).strip()]
            j = i + 1
            while j < len(src) and src[j].startswith('        ') and not re.match(r"\s*if layer==", src[j]):
                body.append(strip_comment(src[j]).strip()); j += 1
            text = ' '.join(b for b in body if b)
            pre, _, ret = text.rpartition('return ')
            parts = top_split(ret)
            return pre.strip().rstrip(';').strip(), parts[0] if parts else ret, (parts[1] if len(parts) > 1 else '')
        return ('', '', '')
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        try: m, x = em.build(case)
        except SystemExit: m, x = None, None
        setup, mexpr, xexpr = line_of(case)
        with torch.no_grad(): y = m.eval()(x) if m is not None else None
        desc = (mexpr or '（源文件中无此 case 的构造，见 hwacha-cc/test/modules 的历史）') + (f'（其中 `{setup}`）' if setup else '')
        nparam = sum(p.numel() for p in m.parameters()) if m is not None else 0
        has_lib = os.path.exists(os.path.join(D, case, 'hwlib.s'))
        t = f'# {case}\n\n'
        t += f'`torch.nn` 单层测试：**{desc}**，一次前向，与 PyTorch 逐元素比对。\n\n'
        if xexpr: t += f'输入：`{xexpr}`' + ('（`x1`/`x4`/`x8`/`x3` 为 1x4x8 / 1x4x8x8 / 1x8x8x8 / 1x4x4x4x4 的随机张量）' if re.fullmatch(r'x[1348]', xexpr) else '') + '。\n\n'
        wrappers = {'Fn': '`Fn(f)`：把一个 functional 调用包成单输入模块', 'Seq0': '`Seq0(m)`：模块返回元组时取第一个元素', 'TwoIn': '`TwoIn(m, shape)`：第二个输入（memory / tgt）固定为随机常量 buffer', 'Unpool': '`Unpool`：先 max-pool（带索引）再 max-unpool', 'MHA': '`MHA`：nn.MultiheadAttention 作自注意力 mha(x,x,x)', 'Emb': '`Emb`：float 输入转 long 索引', 'SDPA': '`SDPA`：qkv 投影 + F.scaled_dot_product_attention + 输出投影', 'Attn': '`Attn`：手写多头自注意力（matmul + softmax）', 'rand_bn': '`rand_bn`：给 BatchNorm 随机的 running 统计量与仿射参数，避免退化'}
        ws = [v for k, v in wrappers.items() if re.search(r'\b' + k + r'\(', mexpr + setup)]
        if ws: t += '包装说明：' + '；'.join(ws) + '。\n\n'
        t += '来源：`hwacha-cc/test/modules/export_module.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重、固定种子。\n\n'
        t += '## 形状\n\n| | 形状 |\n|---|---|\n'
        if x is not None: t += f'| 输入 `x` | {shp(x)} |\n| 输出 | {shp(y)} |\n'
        if nparam: t += f'| 参数量 | {nparam:,} |\n'
        b = bufs_of(m) if m is not None else ''
        if b: t += f'| 常量 buffer | {b} |\n'
        t += '\n## 文件\n\n' + files_table(case, f'{case}.s', [('`mod_main.c`', '通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\\|ref\\|），打印 PASS/FAIL'), (f'`{case}_check.bin`', '输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌')] + ([('`hwlib.s`', '卷积 / 池化库内核（本 case 的汇编调用了它）')] if has_lib else []))
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\n```\n\nhwacha-mlir 映射：`--collapse-all`（所有并行维映射到 lane）。\n\n'
        t += '## Spike 结果\n\n' + fmt_res(R.get(case)) + '\n'
        write(case, t)

elif d == 'torchfunc':
    ef = load(os.path.join(D, 'export_function.py'), 'ef')
    src = open(ef.__file__).read()
    docs = 'https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.%s.html'
    acts_note = {'threshold': 'threshold=0.5, value=0', 'threshold_': 'threshold=0.5, value=0，作用于输入的副本', 'leaky_relu': 'negative_slope=0.1', 'leaky_relu_': 'negative_slope=0.1，作用于输入的副本', 'rrelu': 'training=False（eval 下为固定斜率）', 'rrelu_': 'training=False，作用于输入的副本', 'glu': 'dim=-1', 'softmin': 'dim=-1', 'softmax': 'dim=-1', 'log_softmax': 'dim=-1'}
    rewritten = {'fold': '`F.fold` 会 lower 成 `tm_tensor.scatter`（独立 mlir-opt 无法解析）。这里的 2x2、stride 2 无重叠情形是 unfold 的精确逆运算，导出图用 reshape + permute 实现。',
                 'max_unpool1d': 'max-unpool 的 scatter 是 `tm_tensor` 方言。对同一张量先 max-pool 再 unpool，随机数据无并列极值，结果等于"x 在等于其窗口最大值处保留、其余置零"，导出图用 `x * (x == interpolate(max_pool(x)))` 实现。',
                 'embedding_bag': '`aten.embedding_bag` 在 torch-mlir 中没有 lowering。单个 bag、mean 模式等于各 embedding 的平均，导出图用 `F.embedding(...).mean(1)` 实现。',
                 'pdist': '`aten._pdist_forward` 没有 lowering。导出图用 `index_select` 取出上三角的行对再求范数。',
                 'gaussian_nll_loss': '`F.gaussian_nll_loss` 对 var 的检查是数据相关的，torch.export 无法追踪。导出图用闭式公式 `0.5 * (log(var) + (x - t)^2 / var)` 的均值。',
                 'multilabel_margin_loss': 'torch-mlir 对 `aten.multilabel_margin_loss` 的分解图在 Hwacha 上算出错误值（见 `../known-issues/`）。导出图用闭式公式：目标集合以浮点掩码给出，`mean_b sum_{i in T, j not in T} relu(1 - x_i + x_j) / C`。'}
    rewritten['max_unpool2d'] = rewritten['max_unpool3d'] = rewritten['max_unpool1d']
    def from_source(case):
        """the lambda body of case/rcase/loss('case', lambda s, x: ...) from the source text"""
        m = re.search(r"(?:case|rcase|loss)\('%s',\s*(?:\*unpool\(\d\)|lambda s, x: )" % re.escape(case), src)
        if not m or 'unpool' in m.group(0): return None
        rest = src[m.end():]
        depth, out, q = 0, '', None
        for ch in rest:
            if q: out += ch; q = None if ch == q else q; continue
            if ch in '\'"': q = ch
            elif ch in '([{': depth += 1
            elif ch in ')]}':
                if depth == 0: break
                depth -= 1
            if ch == ',' and depth == 0: break
            out += ch
        return out.strip()
    def body_text(case, m):
        line = inspect.getsource(m.body).strip()
        if case.startswith('max_unpool'):
            d = case[-2]; return f'x * (x == F.interpolate(F.max_pool{d}d(x, 2), scale_factor=2, mode="nearest"))'
        b = from_source(case)
        if b is not None:
            if b.endswith('.reshape(1)') and not b.startswith('('): b = b[:-len('.reshape(1)')]
            return b.replace('s.', '')
        if case in ('threshold', 'threshold_', 'relu', 'relu_', 'hardtanh', 'hardtanh_', 'hardswish', 'relu6', 'elu', 'elu_', 'selu', 'celu', 'leaky_relu', 'leaky_relu_', 'rrelu', 'rrelu_', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish'):
            mm = re.search(r"'%s': (lambda x: )?" % re.escape(case), src)
            if mm:
                rest = src[mm.end():]; depth, out, q = 0, '', None
                for ch in rest:
                    if q: out += ch; q = None if ch == q else q; continue
                    if ch in '\'"': q = ch
                    elif ch in '([{': depth += 1
                    elif ch in ')]}': depth -= 1
                    if (ch == ',' or ch == '}') and depth == 0: break
                    out += ch
                val = out.strip()
                if not mm.group(1): val += '(x)'
            else: val = f'F.{case}(x)'
            return val + (f'　（{acts_note[case]}）' if case in acts_note else '')
        if case.startswith(('dropout', 'alpha_dropout', 'feature_alpha_dropout')): return f'F.{case}(x, 0.5, training=False)　（eval 模式：恒等）'
        mm = re.search(r'lambda s, x: (.*?)(?:,\s*(?:x1|x2|x3|v|idx|lg|theta|R\(|torch\.tensor|\{)|\)\s*$)', line.split('\n')[0])
        b = mm.group(1).rstrip(', ') if mm else line
        if b.endswith('.reshape(1)') and not b.startswith('('): b = b[:-len('.reshape(1)')]
        return b.replace('s.', '')
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        m, x = ef.build(case); m = m.eval()
        with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
        section = next((s for s, names in {'卷积': ['conv', 'unfold', 'fold'], '池化': ['pool'], '注意力': ['attention'], '损失函数': ['loss', 'cross_entropy', 'kl_div', 'nll'], '归一化': ['norm'], 'dropout': ['dropout'], '线性': ['linear', 'bilinear'], '稀疏': ['embedding', 'one_hot'], '距离': ['distance', 'similarity', 'pdist'], '视觉': ['pixel', 'pad', 'interpolate', 'upsample', 'grid', 'affine']}.items() if any(k in case for k in names)), '非线性激活')
        t = f'# {case}\n\n'
        t += f'`torch.nn.functional.{case}`（{section}）的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：{docs % case.rstrip("_")}\n\n'
        t += f'导出图计算：`{body_text(case, m)}`\n\n'
        if case in rewritten: t += f'**注意**：本 case 的导出图与参考不是同一段代码。{rewritten[case]} `check.bin` 中的参考值仍由真正的 `F.{case}` 算出，导出前脚本断言两者一致。\n\n'
        if case.endswith('_'): t += '就地版本：为保持 harness 输入不变，作用在输入的副本上。\n\n'
        if 'loss' in case or case in ('cross_entropy', 'kl_div', 'nll_loss'): t += '损失以 mean 归约为标量，作为 1 元素张量返回。\n\n'
        t += '来源：`export_function.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。\n\n'
        t += '## 形状\n\n| | 形状 |\n|---|---|\n' + f'| 输入 `x` | {shp(x)} |\n| 输出 | {shp(y)} |\n'
        b = bufs_of(m)
        if b: t += f'| 常量 buffer | {b} |\n'
        has_lib = os.path.exists(os.path.join(D, case, 'hwlib.s'))
        t += '\n## 文件\n\n' + files_table(case, f'{case}.s', [('`mod_main.c`', '通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\\|ref\\|），打印 PASS/FAIL'), (f'`{case}_check.bin`', '输入与 PyTorch 参考输出（int32 个数 + float32 数组，各一段），host 用 `.incbin` 内嵌'), ('`HWMLIRFLAGS`', '本 case 使用的 hwacha-mlir 映射选项')] + ([('`hwlib.s`', '卷积 / 池化库内核（本 case 的汇编调用了它）')] if has_lib else []))
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 PyTorch 重新生成\n```\n\n' + f'hwacha-mlir 映射：`{flags(case) or "默认"}`。\n\n'
        t += '## Spike 结果\n\n' + fmt_res(R.get(case)) + '\n'
        write(case, t)

elif d == 'rodinia':
    info = {'nn': ('nearestNeighbor', 'NearestNeighbor', '2048 条记录，每条一个 (纬度, 经度)，计算到查询点的距离', 'N=2048'),
            'kmeans': ('kmeans', 'kmeans_swap（特征矩阵转置）+ kmeans_kernel_c（一步成员分配：每个点找最近的聚类中心）', '1024 个点 x 8 个特征，5 个聚类', 'NP=1024 NF=8 NC=5'),
            'bfs': ('bfs', 'BFS_1 + BFS_2，逐层同步的广度优先搜索，host 像原 OpenCL host 一样迭代到没有新节点', '2048 个节点、出度 4 的随机图，从节点 0 出发', 'NN=2048 DEG=4'),
            'pgain': ('streamcluster', 'memset_kernel + pgain_kernel（对每个点计算打开中心 x 的代价）', '1024 个点 x 8 维，4 个中心', 'NUM=1024 DIM=8 K=4 GROUP=256'),
            'pathfinder': ('pathfinder', 'dynproc_kernel（逐行动态规划，按原 OpenCL host 的方式以 pyramid 高度分块驱动）', '8 行 x 1024 列，pyramid 2，halo 1', 'ROWS=8 COLS=1024 PYRAMID=2 HALO=1 BLOCK=128'),
            'gaussian': ('gaussian', 'Fan1（每个 pivot 的乘子，1-D）+ Fan2（行更新，2-D NDRange），逐 pivot 从 host 驱动，如 gaussianElim.cpp', '64x64 对角占优线性系统', 'N=64 LS=8'),
            'hotspot': ('hotspot', 'hotspot（2-D 热传导 stencil，BLOCK_SIZE x BLOCK_SIZE 分块，每次启动做 pyramid 高度步；hotspot.c 的 compute_tran_temp）', '64x64 网格，4 步，每次启动 2 步；块 8x8（Rodinia 用 16，16x16=256 lane 超过 Hwacha 给这个内核的 maxvl）', 'BS=8 ROWS=64 COLS=64 PYR=2 TOTAL=4'),
            'hotspot3d': ('hotspot3D', 'hotspotOpt1（3-D 热传导 stencil，每个 work-item 负责一根 (x,y) 列并沿 z 扫描；2-D NDRange，如 3D.c）', '32x32x8，2 步，ping-pong', 'NX=32 NY=32 NZ=8 LS=8 ITER=2'),
            'lud': ('lud', 'lud_diagonal + lud_perimeter + lud_internal（分块原地 LU 分解，无 pivoting；每步对角块、周边块、剩余子矩阵，如 lud.cpp）', '64x64 对角占优矩阵，块 8', 'BS=8 DIM=64'),
            'nw': ('nw', 'nw_kernel1 + nw_kernel2（Needleman-Wunsch 动态规划，按 BLOCK_SIZE 块的反对角线推进，如 nw.c）', '64x64 得分矩阵（65x65 含边界），块 16，罚分 10', 'BS=16 N=64 PEN=10'),
            'cfd': ('cfd (euler3d)', 'memset_kernel、initialize_variables、compute_step_factor、compute_flux、time_step：3 阶 RK 的欧拉方程求解，如 euler3d.cpp 的主循环', '256 个单元的合成网格（环形邻接 + 随机邻居，含 wing / far-field 面），2 次迭代', 'NEL=256 ITER=2'),
            'lavamd': ('lavaMD', 'kernel_gpu_opencl（相邻 box 粒子间的 N 体作用力，粒子经 __local 暂存；每个 box 一个 work-group）', '2x2x2=8 个 box，每个 100 个粒子；work-group 64（Rodinia 用 128，超过 maxvl 88）', 'B1D=2 PPB=100 NT=64'),
            'btree': ('b+tree', 'findK（点查询）+ findRangeK（范围查询），B+ 树展平成 knode 数组，每个查询一个 work-group、每个 key 槽一个 lane', 'order 63（Rodinia 256 超过 maxvl），512 个 key，32 个查询', 'ORDER=63 NKEYS=512 NQ=32'),
            'particlefilter': ('particlefilter (naive)', 'particle_kernel（重采样：每个粒子线性扫描 CDF 找第一个 >= u[i] 的项并复制该粒子状态；double）', '512 个粒子，work-group 128', 'NP=512'),
            'leukocyte': ('leukocyte（检测阶段）', 'GICOV_kernel（每个像素在 NCIRCLES 个圆、每圆 NPOINTS 个采样点上计算梯度投影的方差归一化均值的最大值）+ dilate_kernel（strel 窗口内取最大）；buffer 版本', '24x24 像素（梯度图带 MAX_RAD+2 边界），7 圆 x 150 点，strel 25x25', 'W=24 H=24 NPOINTS=150 NCIRCLES=7 MAX_RAD=20 STREL=25'),
            'hybridsort': ('hybridsort（桶排序阶段）', 'histogram1024Kernel（warp-tag 的 __local 原子直方图）、bucketcount（每个元素的桶号与槽位）、bucketprefixoffset、bucketsort（散射）；pivot 与桶起点在 host 上算，如 bucketsort.c', '2048 个 float，1024 个桶；直方图 6144/96、count 与 sort 32-lane 组、prefix 1024/128', 'N=2048 DIVISIONS=1024'),
            'srad': ('srad', 'extract、prepare + reduce（均值 / 方差）、srad_kernel、srad2_kernel、compress，如 kernel_gpu_opencl_wrapper.c', '32x32 图像，2 次迭代；NUMBER_THREADS 64', 'NR=32 NC=32 NITER=2'),
            'backprop': ('backprop', 'bpnn_layerforward_ocl（16x16 work-group，乘积 + 组内树形归约得部分和）+ bpnn_adjust_weights_ocl（权重更新），如 backprop_ocl.cpp', '64 输入单元 -> 16 隐层单元', 'IN=64 HID=16'),
            'myocyte': ('myocyte', 'kernel_gpu_opencl（group 0 / lane 0 跑 ECC 模型，group 1 / lane 0 跑三次 CaM 模型：一次 ODE 右端项求值；两个被调函数标为 always_inline）', '91 个方程，18 个参数', 'EQUATIONS=91 PARAMETERS=18'),
            'dwt2d': ('dwt2d', 'cl_fdwt53Kernel：一级正向 5/3 整数提升小波变换，每个 work-group 处理一个 32x8 的滑动窗口（先列后行，四个象限带输出）；hwacha-cc 以 `-vregs 64` 编译使 32 个 lane 的组落在一个 stripmine 内', '64x64 整数图像', 'SX=64 SY=64 WIN_SX=32 WIN_SY=8'),
            'heartwall': ('heartwall', 'kernel_gpu_opencl：每个 work-group（64 个 lane）跟踪超声心动视频中的一个采样点，第 0 帧提取模板，之后每帧在搜索窗内做归一化互相关（累积和实现）并施加位移掩码；hwacha-cc 以 `-vregs 32` 编译', '51 个点（20 心内膜 + 31 心外膜），3 帧 560x480 合成纹理（每帧平移 1 行 1 列），tSize 5 / sSize 8', 'FRAMES=3 ROWS=560 COLS=480 T_SIZE=5 S_SIZE=8')}
    issues = {}   # the five first-round failures are fixed (../known-issues/README.md keeps the diagnoses)
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        app, kern, size, defs = info.get(case, (case, '', '', ''))
        line = R.get(case, '')
        cyc = re.findall(r'([a-z_0-9]+) (scalar|hwacha-cc): (\d+) cycles', line); verdicts = re.findall(r'([a-z_0-9]+) (PASS|FAIL)', line)
        entries = sorted(set(re.findall(r'^\s*([A-Za-z_0-9]+_ct):', open(os.path.join(D, case, f'{case}.s')).read(), re.M)))
        t = f'# {case}\n\n'
        t += f'Rodinia `{app}` 的 OpenCL 内核在 Hwacha 上运行：**{kern}**。\n\n'
        mod = {'lavamd': '`NUMBER_THREADS` 的 #define 加了 #ifndef 以便由 Makefile 的 -D 覆盖', 'btree': 'kernel_gpu_opencl.cl 与 kernel_gpu_opencl_2.cl 合并为一个文件，第二个文件重复的结构体定义去掉、DEFAULT_ORDER_2 统一为 DEFAULT_ORDER', 'hybridsort': 'histogram1024.cl 与 bucketsort_kernels.cl 合并为一个文件', 'myocyte': 'kernel_ecc / kernel_cam 加了 __attribute__((always_inline))'}.get(case)
        t += f'内核文件 `{case}.cl` 是 Rodinia 3.1 的原版' + (f'，唯一改动：{mod}' if mod else '，未做修改') + f'；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `{"`, `".join(entries)}`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。\n\n'
        t += f'问题规模：{size}（`{case}_main.c` 中 `{defs}`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。\n\n'
        extra = [(f'`{case}.cl`', 'Rodinia 原版 OpenCL 内核' + ('（lavaMD 的 NUMBER_THREADS 改为可由 -D 覆盖）' if case == 'lavamd' else '（两个 .cl 合并为一个文件）' if case in ('btree', 'hybridsort') else '（两个辅助函数加了 always_inline）' if case == 'myocyte' else '')), (f'`{case}_main.c`', '裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数'), ('`common.h`', '伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩')]
        if os.path.exists(os.path.join(D, case, f'{case}_ref.c')): extra.append((f'`{case}_ref.c`', 'Rodinia OpenMP 版的 kernel.c + define.c（仅改函数名）作为标量参考' if case == 'heartwall' else '标量参考：同一 .cl 以 C 编译（OpenCL 限定符定义为空）'))
        if os.path.exists(os.path.join(D, case, f'{case}_ref.h')): extra.append((f'`{case}_ref.h`', '参考实现的 public_struct / private_struct 与入口声明'))
        if os.path.exists(os.path.join(D, case, 'main.h')): extra.append(('`main.h`', 'Rodinia OpenCL 版的 main.h（params_common、NUMBER_THREADS = RD_WG_SIZE）' if case == 'heartwall' else '内核 #include 的 Rodinia host 头文件的替身（fp、NUMBER_THREADS）'))
        t += '## 文件\n\n' + files_table(case, f'{case}.s', extra)
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 .cl 重新生成汇编（clang -> hwacha-cc）\n```\n\n'
        t += '## Spike 结果\n\n'
        if cyc:
            t += '| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |\n|---|---|---|---|\n'
            sc = {k: int(c) for k, w, c in cyc if w == 'scalar'}; sref = next(iter(sc.values()), None)
            for k, w, c in cyc:
                if w == 'hwacha-cc': t += f'| {k} | {sref:,} | {int(c):,} | {sref / int(c):.0f}x |\n' if sref else f'| {k} | | {int(c):,} | |\n'
            t += '\n'
        if verdicts: t += '结果：' + '，'.join(f'{k} **{v}**' for k, v in verdicts) + '。\n'
        if case in issues: t += '\n## 已知问题\n\n' + issues[case] + '\n'
        write(case, t)

elif d == 'polybench':
    # (origin, kernels, problem size, host defines): origin 'gpu' = unmodified PolyBench/GPU 1.0 .cl,
    # 'new' = written for hwacha-cc from the PolyBenchC-4.2.1 loop nest (this suite has no OpenCL version)
    info = {'2dconv': ('gpu', 'Convolution2D_kernel：3x3 卷积模板，每像素一个 work-item', '64x64', 'NI=64 NJ=64'),
            '3dconv': ('gpu', 'Convolution3D_kernel：3-D 模板，每个 i 平面一次 (j, k) 二维启动', '16x32x32', 'NI=16 NJ=32 NK=32'),
            '2mm': ('gpu', 'mm2_kernel1（tmp = alpha*A*B）+ mm2_kernel2（D = tmp*C + beta*D）', '32^4', 'NI=NJ=NK=NL=32'),
            '3mm': ('gpu', 'mm3_kernel1/2/3：E = A*B，F = C*D，G = E*F', '32^5', 'NI=NJ=NK=NL=NM=32'),
            'adi': ('gpu', 'adi_kernel1..6：交替方向隐式求解的行 / 列前代回代（N 在 .cl 中为编译期常量，Makefile 传 -DN=64）', '64x64，2 步', 'N=64 TSTEPS=2'),
            'atax': ('gpu', 'atax_kernel1（tmp = A x）+ atax_kernel2（y = A^T tmp）；内核对 tmp / y 做累加，host 给非零初值', '64x64', 'NX=64 NY=64'),
            'bicg': ('gpu', 'bicgKernel1（q = A p）+ bicgKernel2（s = A^T r）', '64x64', 'NX=64 NY=64'),
            'corr': ('gpu', 'mean_kernel、std_kernel、reduce_kernel、corr_kernel：列均值 / 标准差、中心化归一化、相关矩阵', '40x32', 'M=32 N=40'),
            'covar': ('gpu', 'mean_kernel、reduce_kernel、covar_kernel：列均值、中心化、协方差矩阵', '40x32', 'M=32 N=40'),
            'fdtd-2d': ('gpu', 'fdtd_kernel1/2/3：ey、ex、hz 三步更新，每个时间步各一次二维启动', '48x48，4 步', 'TMAX=4 NX=48 NY=48'),
            'gemm': ('gpu', 'gemm：C = alpha*A*B + beta*C，每个 C 元素一个 work-item', '48^3', 'NI=NJ=NK=48'),
            'gemver': ('gpu', 'gemver_kernel1（A += u1 v1^T + u2 v2^T）、kernel2（x += beta A^T y + z）、kernel3（w += alpha A x）', '64x64', 'N=64'),
            'gesummv': ('gpu', 'gesummv_kernel：y = alpha*A*x + beta*B*x，每行一个 work-item', '64x64', 'N=64'),
            'gramschmidt': ('gpu', 'gramschmidt_kernel1/2/3：修正 Gram-Schmidt QR，k 在 host 迭代', '48x48', 'M=48 N=48'),
            'jacobi-1d': ('gpu', 'runJacobi1D_kernel1（B = 三点平均）+ kernel2（A = B），每步两次启动', '128，8 步', 'N=128 TSTEPS=8'),
            'jacobi-2d': ('gpu', 'runJacobi2D_kernel1（B = 五点平均）+ kernel2（A = B），每步两次二维启动', '64x64，4 步', 'N=64 TSTEPS=4'),
            'lu': ('gpu', 'lu_kernel1（第 k 行归一化）+ lu_kernel2（尾部更新），k 在 host 迭代；PolyBench/GPU 的形式（单位上三角 U），与 4.2.1 的 Doolittle 形式不同', '64x64', 'N=64'),
            'mvt': ('gpu', 'mvt_kernel1（x1 += A y1）+ mvt_kernel2（x2 += A^T y2）', '64x64', 'N=64'),
            'syr2k': ('gpu', 'syr2k_kernel：C = alpha*A*B^T + alpha*B*A^T + beta*C（内核算整个方阵，比对下三角）', '48x48', 'N=48 M=48'),
            'syrk': ('gpu', 'syrk_kernel：C = alpha*A*A^T + beta*C（内核算整个方阵，比对下三角）', '48x48', 'N=48'),
            'symm': ('new', 'symm_kernel：C = alpha*A*B + beta*C，A 对称（存下三角），每个 C 元素一个 work-item，按 kernel_symm 的运算顺序求和', '32x40', 'M=32 N=40'),
            'trmm': ('new', 'trmm_kernel：B = alpha*A^T*B，A 单位下三角；读未修改的 B 写到另一缓冲区', '32x40', 'M=32 N=40'),
            'doitgen': ('new', 'doitgen_kernel：A[r][q][:] = A[r][q][:]*C4，每个 (r, q) 一个 work-item，sum 用全局暂存区', '16x16x16', 'NR=NQ=NP=16'),
            'cholesky': ('new', 'cholesky_kernel1/2/3：右视 Cholesky（列缩放、对角开方、尾部更新），k 在 host 迭代；与 4.2.1 的行视形式做同样的减法、同样的顺序，结果精确', '48x48', 'N=48'),
            'durbin': ('new', 'durbin_kernel1/2/3：Levinson-Durbin 递推，每步 k 三次启动（单 work-item 推进 alpha/beta/sum，k 个 work-item 算 z 再拷回）', '64', 'N=64'),
            'ludcmp': ('new', 'ludcmp_kernel1/2（右视 Doolittle LU，k 在 host）+ kernel3/4（按列的前代、回代，每行一次启动）', '48x48', 'N=48'),
            'trisolv': ('new', 'trisolv_kernel：按列的前代（第 i 行的 work-item 完成 x[i]，其余行减去第 i 列），每行一次启动；减法顺序与 4.2.1 相同，结果精确', '64x64', 'N=64'),
            'deriche': ('new', 'deriche_kernel1..5：Deriche 递归高斯边缘滤波的四遍 IIR（每行 / 每列一个 work-item）与两次合成（每像素一个）；系数在 host 用 expf/powf 算好传入', '32x32', 'W=32 H=32'),
            'floyd-warshall': ('new', 'floyd_warshall_kernel：全源最短路，每个中间点 k 一次二维启动（int）', '48x48', 'N=48'),
            'nussinov': ('new', 'nussinov_kernel：RNA 二级结构打分表，按对角线 d = j - i 逐条启动，对角线上每个元素一个 work-item（int）', '48', 'N=48'),
            'heat-3d': ('new', 'heat_3d_kernel：7 点三维热传导模板，每个内部 (i, j) 一个 work-item、k 在 lane 内循环，每步两次启动（A->B、B->A）', '16^3，4 步', 'N=16 TSTEPS=4'),
            'seidel-2d': ('new', 'seidel_2d_kernel：就地 Gauss-Seidel 九点平均，按反对角线 i + j 逐条启动（每条对角线上的元素相互独立）', '32x32，2 步', 'N=32 TSTEPS=2')}
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        origin, kern, size, defs = info.get(case, ('gpu', '', '', ''))
        line = R.get(case, '')
        cyc = re.findall(r'([a-z_0-9-]+) (scalar|hwacha-cc): (\d+) cycles', line); verdicts = re.findall(r'([a-z_0-9-]+) (PASS|FAIL)', line)
        entries = sorted(set(re.findall(r'^\s*([A-Za-z_0-9]+_ct):', open(os.path.join(D, case, f'{case}.s')).read(), re.M)))
        t = f'# {case}\n\n'
        t += f'PolyBench `{case}` 在 Hwacha 上运行：**{kern}**。\n\n'
        if origin == 'gpu':
            t += f'内核文件 `{case}.cl` 是 PolyBench/GPU 1.0 的原版 OpenCL 内核，未做修改'
        else:
            t += f'PolyBench/GPU 没有这个 case 的 OpenCL 版本：`{case}.cl` 是按 PolyBenchC-4.2.1 的 `kernel_{case.replace("-", "_")}` 循环嵌套为 hwacha-cc 改写的 OpenCL 内核，保持原公式与浮点运算顺序'
        t += f'；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 ' + '、'.join(f'`{e}`' for e in entries) + f'，host 按原 host 的顺序（依赖型算法把外层循环留在 host，每次迭代启动一个小内核）调用。\n\n'
        t += f'问题规模：{size}（`{case}_main.c` 中 `{defs}`），输入与标量按 PolyBenchC-4.2.1 的 `init_array`。host 先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印 PASS/FAIL 与周期数。\n\n'
        extra = [(f'`{case}.cl`', 'PolyBench/GPU 原版 OpenCL 内核' if origin == 'gpu' else '按 PolyBenchC-4.2.1 改写的 OpenCL 内核'),
                 (f'`{case}_main.c`', '裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数'),
                 ('`common.h`', '伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩')]
        t += '## 文件\n\n' + files_table(case, f'{case}.s', extra)
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 .cl 重新生成汇编（需要 hwacha-cc）\n```\n\n'
        t += '## Spike 结果\n\n'
        if cyc:
            t += '| 内核 | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |\n|---|---|---|---|\n'
            sc = {k: int(c) for k, w, c in cyc if w == 'scalar'}; sref = next(iter(sc.values()), None)
            for k, w, c in cyc:
                if w == 'hwacha-cc': t += f'| {k} | {sref:,} | {int(c):,} | {sref / int(c):.0f}x |\n' if sref else f'| {k} | | {int(c):,} | |\n'
            t += '\n'
        if verdicts: t += '结果：' + '，'.join(f'{k} **{v}**' for k, v in verdicts) + '。\n'
        write(case, t)

elif d == 'deepbench':
    info = {'gemm': ('GEMM', 'gemm_bench：cublasSgemm 语义的单精度 GEMM，列主序，问题集里的 NN / TN / NT 三种转置组合各一个内核（gemm_nn / gemm_tn / gemm_nt），每个 C 元素一个 work-item', 'training 与 inference 集的 7 个形状，缩小 16–128 倍'),
            'gemm-int8': ('GEMM（int8 推理）', 'gemm_bench inference int8：cublasGemmEx CUDA_R_8I x CUDA_R_8I -> CUDA_R_32I，内核 gemm_i8 做 int8 乘、int32 累加，比对精确', 'inference server / device 集的 6 个形状，缩小 16–64 倍'),
            'conv': ('卷积', 'conv_bench：cuDNN 的前向（conv_fwd）、反向数据（conv_bwd_data）、反向权重（conv_bwd_filter），NCHW、互相关；直接卷积形式，每个输出元素一个 work-item，归约循环在 lane 内', 'training 集中 VGG / ResNet / Inception / DeepSpeech 的 5 个层，通道与批缩小'),
            'rnn-vanilla': ('vanilla RNN', 'rnn_bench "vanilla"：cuDNN CUDNN_RNN_RELU、单层单向、CUDNN_SKIP_INPUT（输入直接进单元，无输入权重矩阵）；h_t = ReLU(x_t + R h_{t-1} + b)，每个时间步一次启动，每个 (batch, unit) 一个 work-item', 'training 集的 3 个形状（隐层 1760–2560）缩小'),
            'rnn-lstm': ('LSTM', 'rnn_bench "lstm"：cuDNN CUDNN_LSTM（门序 i, f, o, g）、单层单向、SKIP_INPUT；每个 work-item 算自己单元的四个门点积，sigmoid / tanh 用 exp 展开', 'training 集的 3 个形状（隐层 512–2048）缩小'),
            'rnn-gru': ('GRU', 'rnn_bench "gru"：cuDNN CUDNN_GRU（门序 r, z, h，h\' = tanh(x + r * (R_h h + b_Rh) + b_Wh)）、单层单向、SKIP_INPUT', 'training 集的 3 个形状（隐层 1024–2816）缩小'),
            'gemm-fp16': ('GEMM（fp16）', 'gemm_bench "half" 精度：cublasGemmEx 16F 输入输出、32F 计算（DeepBench 的 "FP16 inputs / FP32 math"）；内核 gemm_nn / gemm_tn / gemm_nt 用 vlxh + vfcvt.s.h 装载、vfmadd.s 累加、vfcvt.h.s 一次舍入写回；gemm_nn_h 是纯半精度算术（vfmadd.h，每步一次舍入），参考按同样的舍入建模，全部精确', '与 gemm 相同的 6 个形状，另加 2 个纯半精度算术'),
            'conv-fp16': ('卷积（fp16）', 'conv_bench "half" 精度：CUDNN_DATA_HALF 张量、float 计算；与 conv 相同的三个方向，半精度装载、单精度累加、写回时一次舍入', '与 conv 相同的 5 个层'),
            'rnn-lstm-fp16': ('LSTM（fp16）', 'rnn_bench "lstm" 的 "half" 精度：x / R / b / h / c 都是 half，门运算在 float 中进行，h_t / c_t 以 half 存回（下一步读回时已量化，与 cuDNN 的 half 状态一致）', '与 rnn-lstm 相同的 3 个形状'),
            'rnn-vanilla-fp16': ('vanilla RNN（fp16）', 'rnn_bench "vanilla" 的 "half" 精度：x / R / b / h 都是 half，ReLU 单元的运算在 float 中进行，h_t 以 half 存回；比对精确', '与 rnn-vanilla 相同的 3 个形状'),
            'rnn-gru-fp16': ('GRU（fp16）', 'rnn_bench "gru" 的 "half" 精度：x / R / bW / bR / h 都是 half，门运算在 float 中进行，h_t 以 half 存回', '与 rnn-gru 相同的 3 个形状'),
            'sparse-gemm': ('稀疏 GEMM', 'sparse_bench：cusparseScsrmm，A 为 CSR（稀疏度 0.9 / 0.95，按 DeepBench 的方式由均匀随机数阈值化生成）、B 稠密，alpha = 1/k、beta = 0；内核 csrmm 每个 C 元素一个 work-item，沿行的非零元循环', 'inference server / device 集的 5 个形状缩小 32–64 倍')}
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        title, kern, size = info.get(case, (case, '', ''))
        line = R.get(case, '')
        runs = re.findall(r'(?:^|\s)' + re.escape(case) + r' (.*?): scalar (\d+) cycles, hwacha-cc (\d+) cycles', line)
        verdicts = re.findall(r'(?:^|\s)(' + re.escape(case) + r') (PASS|FAIL)', line)
        entries = sorted(set(re.findall(r'^\s*([A-Za-z_0-9]+_ct):', open(os.path.join(D, case, f'{case}.s')).read(), re.M)))
        t = f'# {case}\n\n'
        t += f'DeepBench 的 **{title}** 在 Hwacha 上运行：{kern}。\n\n'
        t += f'DeepBench（baidu-research/DeepBench）本身没有内核，它在 `code/kernels/*.h` 的问题集上调用厂商库（cuDNN / cuBLAS / cuSPARSE / MKL ...）；`{case}.cl` 是按该库的语义为 hwacha-cc 写的 OpenCL 实现。hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 ' + '、'.join(f'`{e}`' for e in entries) + '。\n\n'
        t += f'问题规模：{size}（`{case}_main.c` 的 `shapes[]`，每项注明对应的 DeepBench 原形状与缩放比）。输入由固定种子的伪随机数生成；host 对每个形状先在 Rocket 标量核上跑 C 参考实现，再跑 Hwacha 内核，逐元素比对并打印两侧周期数。\n\n'
        extra = [(f'`{case}.cl`', '按 DeepBench 所调用库的语义写的 OpenCL 内核'),
                 (f'`{case}_main.c`', '裸机 host：问题形状表、输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数'),
                 ('`common.h`', '伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、percentDiff、__errno 与陷阱桩')]
        t += '## 文件\n\n' + files_table(case, f'{case}.s', extra)
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 .cl 重新生成汇编（需要 hwacha-cc）\n```\n\n'
        t += '## Spike 结果\n\n'
        if runs:
            t += '| 形状（DeepBench 原形状与缩放） | 标量 Rocket 周期 | Hwacha 周期 | 加速比 |\n|---|---|---|---|\n'
            for shape, sc, hw in runs: t += f'| {shape} | {int(sc):,} | {int(hw):,} | {int(sc) / int(hw):.0f}x |\n'
            t += '\n'
        if verdicts: t += '结果：' + '，'.join(f'{k} **{v}**' for k, v in verdicts) + '。\n'
        write(case, t)

elif d == 'shoc':
    # (SHOC level, kernel description, problem size, notes)
    info = {'triad': ('level1', 'Triad（Triad.cpp 内嵌的内核）：C = A + s*B，每元素一个 work-item，s = 1.75', '64 / 128 / 256 KB 三种块大小（SHOC 到 16 MB）', ''),
            'reduction': ('level1', 'reduce（每个 work-group 把跨步切片累加进 __local，树形归约后写一个部分和，host 相加）+ reduceNoLocal（SHOC 给工作组大小为 1 的设备的备用内核）', '16384 个 float（i % 3），64 个 work-group', ''),
            'scan': ('level1', 'reduce + top_scan + bottom_scan：三段前缀和（每组区域求和、单组扫描组和、每组以组和为种子按 4 元素向量扫描），按 Scan.cpp 的顺序启动', '16384 个 float，64 个 work-group x 64', ''),
            'sort': ('level1', 'reduce + top_scan + bottom_scan：LSD 基数排序，4 位数字、8 趟，每趟三次启动，两个缓冲区乒乓', '16384 个 uint（i % 16），64 个 work-group x 32', 'clang 以 -Xclang -disable-llvm-passes 编译：-O2 的 GlobalOpt 会把 top_scan 的 __local int s_seed 变成每 lane 的私有值'),
            'spmv': ('level1', 'spmv_csr_scalar_kernel（每行一个 work-item）、spmv_csr_vector_kernel（每行 vecWidth 个 work-item，__local 部分和树形归约）、spmv_ellpackr_kernel（列主序 ELLPACK-R）', '256 x 256、1% 非零（util.h initRandomMatrix），值与 x 均匀分布于 [0, 10)', ''),
            'md': ('level1', 'compute_lj_force：Lennard-Jones 力，每原子一个 work-item，邻居表转置存放（neighList[j*inum + idx]）；邻居表按 MD.cpp 的方式取最近的 maxNeighbors 个原子', '512 个原子（SHOC 12288），128 个邻居，cutsq 16、lj1 1.5、lj2 2.0', 'float4 位置 / 力由 hwacha-cc 的 scalarizer 拆成标量'),
            'md5hash': ('level1', 'FindKeyWithDigest_Kernel：在密钥空间里暴力搜索 MD5 摘要等于目标的密钥，每个 work-item 处理 valsPerByte 个连续密钥', '4 字节 x 10 个取值 = 10^4 个密钥（SHOC 7 x 10），3 个随机目标', 'clang 以 -Dinline=static 编译保留 md5_2words 的定义；LEFTROTATE 形成的 llvm.fshl 由 hwacha-cc 展开'),
            'stencil2d': ('level1', 'StencilKernel（每个 work-item 从 __local 的带 halo 分块算 LROWS 行）+ CopyRect（把左右 halo 列搬到新缓冲区），每次迭代交换缓冲区', '66 x 66（64 x 64 内部），10 次迭代（SHOC 1000），权重 0.25 / 0.15 / 0.05', ''),
            'bfs': ('level1', 'BFS_kernel_warp（bfs_iiit.cl）：逐层同步 BFS，每个 warp 扫 CHUNK_SZ 个顶点、按 lane 展开邻居，flag 告知 host 是否还有下一层；bfs_uiuc_spill.cl 的 5 个内核也编进了 bfs.s，但未写 host', '2048 个顶点的 GenerateSimpleKWayGraph（度 2），源点 0', 'hwacha-cc 新增 get_num_groups(0)'),
            'fft': ('level1', 'fft1D_512 / ifft1D_512（64 个 work-item 一组做一个 512 点复数 FFT：基 8 三遍、旋转因子、两次经 __local 的转置）+ chk1D_512（SHOC 的自检内核）', '8 个 512 点 FFT（SHOC 256 个），后半批是前半批的副本', 'float2 由 scalarizer 拆开；旋转因子的 sin / cos 由 hwacha-cc 新增的展开实现'),
            'gemm': ('level1', 'sgemmNN / sgemmNT（源自 MAGMA：16 x 4 的 work-group 算 64 x 16 的 C 分块，A 每 work-item 暂存 4 个元素，B 经 __local 16 x 17 分块），alpha = 1、beta = -1', '128 x 128（SHOC 256），输入均匀分布于 [0.5, 2)', '内核要求 64 lane 的 work-group（分块映射写死），在 Hwacha 上即 <= 32 个向量寄存器，以 -vregs 32 生成；k 循环里 16 个 C 累加器 + 4 个 A 值 + 地址长期活跃，靠 hwacha-cc 的累加链原地计算、gather 偏移共享、uniform 出口值别名与溢出器放进 32 个寄存器'),
            's3d': ('level2', '27 个内核（gr_base、ratt..ratt10、rdsmh、ratx、ratxb、ratx2、ratx4、qssa、qssab、qssa2、rdwdot..rdwdot10）按 S3D.cpp 的两阶段顺序启动：22 种组分、206 个反应的化学动力学右端项（组分生成率 WDOT）', '64 个网格点（N_GP 编进内核，-DN_GP=64）；p 1.0132e6、T 1000、y 按 S3D.cpp', 'SHOC 不校验 S3D；这里把同一批 .cl 编成 C（s3d_ref.c）做参考，全部 6 个输出数组相对误差 < 1e-6；exp10 由 hwacha-cc 新增的展开实现')}
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        level, kern, size, note = info.get(case, ('', '', '', ''))
        line = R.get(case, '')
        cyc = []   # every "<label>: N cycles" / "<label> N cycles" the host printed (the hosts use several formats)
        for m in re.finditer(r'(\d+) cycles', line):
            pre = re.sub(r'\d+\.\d+ cyc/elem', '|', line[:m.start()])
            segs = [x.strip().rstrip(':').strip() for x in re.split(r'[,|()]', pre) if x.strip()]
            lab = re.sub(r'^\d+ mismatches\s*', '', segs[-1]) if segs else ''
            if case not in lab:
                ctx = next((x for x in reversed(segs) if case in x), '')
                if ctx: lab = re.sub(r'^\d+ mismatches\s*', '', ctx.split(':')[0].strip()) + ' / ' + lab
            cyc.append((lab[:80], int(m.group(1))))
        verdicts = re.findall(r'(?:^|\s)(' + re.escape(case) + r') (PASS|FAIL)', line)
        entries = sorted(set(re.findall(r'^\s*([A-Za-z_0-9]+_ct):', open(os.path.join(D, case, f'{case}.s')).read(), re.M)))
        cls = sorted(f for f in os.listdir(os.path.join(D, case)) if f.endswith('.cl'))
        t = f'# {case}\n\n'
        t += f'SHOC {level} `{case}` 在 Hwacha 上运行：**{kern}**。\n\n'
        t += f'内核文件（' + '、'.join(f'`{c}`' for c in cls) + '）是 SHOC 的原版，未做修改（多文件的基准由一个同名 .cl #include 汇总）；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 ' + '、'.join(f'`{e}`' for e in entries) + f'，host 按 SHOC host 的启动顺序与工作组形状调用（__local 缓冲区是 host 数组：每个 work-group 独占一个 stripmine）。\n\n'
        t += f'问题规模：{size}（`{case}_main.c`）。输入按 SHOC host 的方式生成（固定种子）；host 先在 Rocket 标量核上跑参考实现，再跑 Hwacha 内核，按 SHOC 的判据比对并打印 PASS/FAIL 与周期数。\n\n'
        if note: t += f'说明：{note}。\n\n'
        extra = [(f'`{c}`', 'SHOC 原版 OpenCL 内核' + ('（汇总 #include）' if c == f'{case}.cl' and len(cls) > 1 else '')) for c in cls]
        extra += [(f'`{case}_main.c`', '裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数'),
                  ('`common.h`', '伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩')]
        if os.path.exists(os.path.join(D, case, f'{case}_ref.c')): extra.append((f'`{case}_ref.c`', '标量参考' + ('：同一批 .cl 编成 C' if case == 's3d' else '：SHOC host 侧的 md5_2words / IndexToKey')))
        if os.path.exists(os.path.join(D, case, f'{case}_ref.h')): extra.append((f'`{case}_ref.h`', '参考实现的声明'))
        t += '## 文件\n\n' + files_table(case, f'{case}.s', extra)
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 .cl 重新生成汇编（需要 hwacha-cc）\n```\n\n'
        t += '## Spike 结果\n\n'
        if cyc:
            t += '| 项 | 周期 |\n|---|---|\n'
            for lab, c in cyc: t += f'| {lab} | {c:,} |\n'
            t += '\n'
        if verdicts: t += '结果：' + '，'.join(f'{k} **{v}**' for k, v in verdicts) + '。\n'
        write(case, t)

elif d == 'deformable':
    sys.path.insert(0, D); import export_dcn as E
    desc = {'deform_conv': ('一个 3x3 可变形卷积（DCN v1）：偏移量由同一输入上的 3x3 卷积预测，每个采样点双线性插值后做 im2col 矩阵乘', '输出特征图 1x64x16x16'),
            'deform_conv_v2': ('一个 3x3 可变形卷积（DCN v2）：偏移量加 sigmoid 调制掩码', '输出特征图 1x64x16x16'),
            'psroi_pool': ('位置敏感 RoI 池化（R-FCN），k=7，3 个固定 RoI；bin 规则与 torchvision.ops.ps_roi_pool 完全一致', '池化结果 (3, 4, 7, 7)'),
            'deform_psroi': ('可变形位置敏感 RoI 池化：先做一次常规 PS-RoI 池化，3x3 卷积预测每个 bin 的 (dy, dx)，再按偏移双线性采样', '池化结果 (3, 4, 7, 7)'),
            'deeplab': ('DeepLab（仓库 deeplab/ 的形式）：ResNet-101 conv5 空洞化（输出 stride 16）+ 1x1 分类器 + 双线性上采样到输入尺寸，Cityscapes 19 类', 'logits 图 1x19x64x64'),
            'rfcn': ('R-FCN：ResNet-101 conv5 空洞化 -> 1x1 降维到 1024 -> 位置敏感的类别得分图（k*k*C）与 bbox 图（k*k*4）-> 3 个固定 RoI 的 PS-RoI 池化 -> 各 bin 平均投票；conv5 上另有 RPN 头', 'RPN 的 cls/box 图、类别投票、bbox 投票拼接成一行'),
            'rcnn': ('Faster R-CNN（2fc 头）：ResNet-101 conv5 空洞化 -> 1x1 降维到 256 -> 3 个固定 RoI 的 7x7 平均 RoI 池化 -> fc1024 x2 -> cls / box；conv5 上另有 RPN 头', 'RPN 的 cls/box 图、cls 与 box 得分拼接成一行'),
            'fpn': ('FPN：ResNet-101 的 C2-C5 经 1x1 侧向连接 + 自顶向下 + 3x3 平滑得到 P2-P5，P6 为 P5 的 stride-2 池化；共享 RPN 头作用于每一级', '各级 RPN 头输出拼接成一行')}
    dcn_note = '`_dcn` 变体：conv5 阶段（res5a-c）的 3x3 卷积换成可变形卷积（论文的做法）'
    extra = {'rfcn': '，RoI 池化换成可变形 PS-RoI 池化（每个 bin 的偏移由第一次池化的特征经 3x3 卷积预测，gamma=0.1）', 'rcnn': '，RoI 池化换成可变形 RoI 池化（每个 bin 的偏移由 fc 预测）', 'fpn': '', 'deeplab': ''}
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}_tv.s')): continue
        base = re.sub(r'_(dcn|voc|coco|v2)', '', case) if case not in desc else case
        if case in ('deform_conv_v2', 'deform_psroi'): base = case
        d0, out = desc.get(base, ('', ''))
        ncls = {'_voc': 'VOC 21 类', '_coco': 'COCO 81 类'}
        cls_note = next((v for k, v in ncls.items() if case.endswith(k)), '')
        if base == 'deeplab' and case.endswith('_voc'): d0 = d0.replace('Cityscapes 19 类', 'VOC 21 类'); out = out.replace('19', '21')
        ctor, shape = E.MODELS[case]; torch.manual_seed(0); m = ctor().eval()
        nparam = sum(p.numel() for p in m.parameters()) / 1e6
        cb = os.path.join(D, case, f'{case}_tv_check.bin'); nin = nout = None
        if os.path.exists(cb):
            with open(cb, 'rb') as f: f.read(4); nin = struct.unpack('i', f.read(4))[0]; f.seek(8 + nin * 4); nout = struct.unpack('i', f.read(4))[0]
        wpath = os.path.join(D, case, f'{case}_tv_weights.bin'); wsz = os.path.getsize(wpath) if os.path.exists(wpath) else None
        t = f'# {case}\n\n'
        t += f'Deformable ConvNets（github.com/msracver/Deformable-ConvNets）的 **{case}** 在 Hwacha 上的一次前向，与 PyTorch 比对。\n\n'
        t += f'模型：{d0}' + (f'（{cls_note}）' if cls_note and base != 'deeplab' else '') + '。\n\n'
        if '_dcn' in case: t += f'{dcn_note}{extra.get(base, "")}。\n\n'
        t += f'比对内容：{out}，逐元素比对（容差 1e-4 + 1e-2·max\\|ref\\|）。\n\n'
        t += '权重随机（固定种子，未加载 MXNet 的 .params；BatchNorm 给随机 running 统计量），输入随机，64x64 图像；检测模型的 RoI 固定、RPN 头输出计入比对，proposal 选择 / NMS 因形状数据相关未导出。可变形卷积与 PS-RoI 池化是 `../dcn_ops.py` 中的纯张量实现（torch-mlir 无法 lower torchvision 的自定义算子），与 torchvision.ops 数值一致到 1e-7。\n\n'
        t += '## 形状与规模\n\n| | 值 |\n|---|---|\n' + f'| 输入 | {"x".join(map(str, shape))}' + (f'（{nin:,} 个 float）' if nin else '') + ' |\n'
        if nout: t += f'| 输出元素数 | {nout:,} |\n'
        t += f'| 参数量 | {nparam:.1f}M |\n'
        if wsz: t += f'| 权重 blob | {wsz / 1048576:.0f} MB |\n'
        t += '\n## 文件\n\n' + files_table(case, f'{case}_tv.s', [(f'`{case}_tv_weights.bin.S`', '权重的 `.incbin` 桩（`split_weights.py` 分成 `.weights_lo` / `.weights_hi` 两段）'), (f'`{case}_tv_weights.bin`', f'权重 blob（不入 git，`make gen-{case}` 按固定种子重建）'), (f'`{case}_tv_check.bin`', '输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌'), ('`tv_main.c`', '通用 host：调用 `net`，比对 max\\|diff\\|（分类型输出还比对 argmax）'), ('`hwlib.s`', '卷积 / 池化库内核')] + ([('`HWMLIRFLAGS`', '本 case 需要的 hwacha-mlir 映射选项')] if flags(case) else []))
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '_tv.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 PyTorch 重新生成\n```\n\n' + f'hwacha-mlir 映射：`{flags(case) or "默认（最内维为 lane）"}`。\n\n'
        t += '## Spike 结果\n\n' + fmt_res(R.get(case)) + '\n'
        write(case, t)

elif d == 'torchintf':
    ef = load(os.path.join(D, 'export_intf.py'), 'ef')
    docs = 'https://docs.pytorch.org/docs/2.14/generated/torch.%s.html'
    calls = {'topk': ('torch.topk(x, 4)', 'dim=-1、largest=True、sorted=True：每行最大的 4 个值（降序）及其下标，输出为每行 [values | indices]'),
             'fft': ('torch.fft.fft(x)', '4 行、长度 16 的实数序列，输出每行 [实部 | 虚部]'), 'ifft': ('torch.fft.ifft(x)', '输入每行 [实部 | 虚部] 各 16，输出同布局'),
             'rfft': ('torch.fft.rfft(x)', '实数序列长 16，输出 9 个半谱 bin 的 [实部 | 虚部]'), 'irfft': ('torch.fft.irfft(x, 16)', '输入 9 个 bin 的 [实部 | 虚部]，输出长 16 的实数序列'),
             'hfft': ('torch.fft.hfft(x, 16)', '厄米对称的单侧输入（9 个 bin 的 [实部 | 虚部]），输出长 16 的实数序列'), 'ihfft': ('torch.fft.ihfft(x)', '实数序列长 16，输出 9 个 bin 的 [实部 | 虚部]'),
             'fft2': ('torch.fft.fft2(x)', '2 个 8x8 实数图，最后两维变换，输出 [实部 | 虚部]'), 'ifft2': ('torch.fft.ifft2(x)', '输入 2 x 8 x [8 | 8]，输出同布局'),
             'rfft2': ('torch.fft.rfft2(x)', '2 个 8x8 实数图，输出 2 x 8 x [5 | 5]'), 'irfft2': ('torch.fft.irfft2(x, s=(8, 8))', '输入 2 x 8 x [5 | 5]，输出 2 个 8x8 实数图'),
             'hfft2': ('torch.fft.hfft2(x, s=(8, 8))', '输入 2 x 8 x [5 | 5]（最后一维单侧），输出 2 个 8x8 实数图'), 'ihfft2': ('torch.fft.ihfft2(x)', '2 个 8x8 实数图，输出 2 x 8 x [5 | 5]'),
             'fftn': ('torch.fft.fftn(x)', '4x4x4 实数张量，三维全变换，输出 4 x 4 x [4 | 4]'), 'ifftn': ('torch.fft.ifftn(x)', '输入 4 x 4 x [4 | 4]，输出同布局'),
             'rfftn': ('torch.fft.rfftn(x)', '4x4x4 实数张量，输出 4 x 4 x [3 | 3]'), 'irfftn': ('torch.fft.irfftn(x, s=(4, 4, 4))', '输入 4 x 4 x [3 | 3]，输出 4x4x4 实数张量'),
             'hfftn': ('torch.fft.hfftn(x, s=(4, 4, 4))', '输入 4 x 4 x [3 | 3]，输出 4x4x4 实数张量'), 'ihfftn': ('torch.fft.ihfftn(x)', '4x4x4 实数张量，输出 4 x 4 x [3 | 3]'),
             'fftfreq': ('x + torch.fft.fftfreq(16)', '频率向量是常量，加到长 16 的输入上'), 'rfftfreq': ('x + torch.fft.rfftfreq(16)', '频率向量是常量，加到长 9 的输入上'),
             'fftshift': ('torch.fft.fftshift(x)', '5x7 张量，两个维度都移位'), 'ifftshift': ('torch.fft.ifftshift(x)', '5x7 张量，两个维度都移位（奇数尺寸下与 fftshift 不同）')}
    fftnote = ('torch-mlir 没有复数张量：复数张量表示为实数张量，实部与虚部沿最后一维拼接（`[实部 | 虚部]`），实数输入 / 输出保持原形状。沿某一维的 DFT 是与常量块矩阵的矩阵乘：'
               '沿最后一维 `[xr | xi] @ [[Fr, Fi], [-Fi, Fr]]`（实数输入 `x @ [Fr | Fi]`）；沿其他维 `Fr @ x + Fi @ (x @ P)`，`P = [[0, I], [-I, 0]]` 把 `[xr | xi]` 变成 `[-xi | xr]`；'
               'F = C - iS（正变换）或 (C + iS) / N（逆变换），C、S = cos / sin(2πnk/N)。rfft 取前 N/2+1 列；irfft 是带厄米权重 (1, 2, ..., 2, 1) / N 的 C2R 求和；'
               'hfft(x) = N·irfft(conj x)，ihfft(x) = conj(rfft x) / N（n 维同理，N 为各维长度之积）。矩阵在 float64 里生成再转 float32。')
    rewritten = {f: fftnote for f in ('fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'fft2', 'ifft2', 'rfft2', 'irfft2', 'hfft2', 'ihfft2', 'fftn', 'ifftn', 'rfftn', 'irfftn', 'hfftn', 'ihfftn')}
    wincalls = {'bartlett': 'bartlett(16)', 'blackman': 'blackman(16)', 'cosine': 'cosine(16)', 'exponential': 'exponential(16, tau=3.0)', 'gaussian': 'gaussian(16, std=3.0)',
                'general_cosine': 'general_cosine(16, a=[0.42, 0.5, 0.08])', 'general_hamming': 'general_hamming(16, alpha=0.6)', 'hamming': 'hamming(16)', 'hann': 'hann(16)',
                'kaiser': 'kaiser(16, beta=12.0)', 'nuttall': 'nuttall(16)'}
    for w, c in wincalls.items(): calls[w] = (f'x * torch.signal.windows.{c}', '给 4 行、长 16 的信号加窗；窗函数没有张量输入，窗在导出图里按定义计算（cos / sin / exp / pow / abs 在 Hwacha 上求值）')
    rewritten['kaiser'] = '`torch.i0`（零阶修正贝塞尔函数）在 torch-mlir 中没有 lowering。导出图用幂级数 I0(z) = Σ_k (z/2)^{2k} / (k!)^2 按 Horner 法则算 30 项（beta = 12 时 z/2 <= 6，float32 下与 torch.i0 相差 < 1e-6），分母 I0(beta) 是常量。'
    lacalls = {'norm': ('torch.linalg.norm(x)', '4x4 矩阵的 Frobenius 范数'), 'vector_norm': ('torch.linalg.vector_norm(x)', '4x4 矩阵全部元素的 2 范数'),
               'matrix_norm': ('torch.linalg.matrix_norm(x)', '默认 Frobenius 范数'), 'diagonal': ('torch.linalg.diagonal(x)', '4x4 矩阵的对角线'),
               'det': ('torch.linalg.det(x)', '4x4 对角占优矩阵'), 'slogdet': ('torch.linalg.slogdet(x)', '输出 [sign, logabsdet]'),
               'cond': ('torch.linalg.cond(x)', '2 范数条件数 = σmax / σmin'), 'matrix_rank': ('torch.linalg.matrix_rank(x, rtol=1e-3)', '6x4、秩 3 的矩阵，输出 3'),
               'cholesky': ('torch.linalg.cholesky(x)', '4x4 对称正定矩阵，输出下三角 L'), 'qr': ('torch.linalg.qr(x)', '输出 [Q | R]（4x8）'),
               'lu': ('torch.linalg.lu(x)', '输出 [L | U]（4x8），P 为单位置换'), 'lu_factor': ('torch.linalg.lu_factor(x)', '输出紧凑的 LU（L 的严格下三角 + U）'),
               'eigh': ('torch.linalg.eigh(x)', '4x4 对称矩阵，输出 [V | w]（4x5），特征值升序，特征向量符号按第一分量归一'), 'eigvalsh': ('torch.linalg.eigvalsh(x)', '升序特征值'),
               'eig': ('torch.linalg.eig(x)', '对称输入（实特征值），输出 [V | Re w]（4x5），按实部升序、符号按第一分量归一'), 'eigvals': ('torch.linalg.eigvals(x)', '对称输入，特征值实部升序'),
               'svd': ('torch.linalg.svd(x)', '输出 [U | σ | Vh]（4x9），σ 降序，V 的列符号按第一分量归一（U 随之）'), 'svdvals': ('torch.linalg.svdvals(x)', '降序奇异值'),
               'solve': ('torch.linalg.solve(x, B)', 'B 为 4x2 常量'), 'solve_triangular': ('torch.linalg.solve_triangular(x, B, upper=True)', '输入上三角矩阵，B 为 4x2 常量'),
               'lu_solve': ('torch.linalg.lu_solve(LU, pivots, x)', '输入为右端项 B（4x2），LU 与 pivots 是常量（lu_factor 的结果）'),
               'lstsq': ('torch.linalg.lstsq(x, B).solution', '6x4 满列秩矩阵，B 为 6x3 常量'), 'inv': ('torch.linalg.inv(x)', '4x4 对角占优矩阵'),
               'pinv': ('torch.linalg.pinv(x)', '6x4 满列秩矩阵'), 'matrix_exp': ('torch.linalg.matrix_exp(x)', '4x4 矩阵'), 'matrix_power': ('torch.linalg.matrix_power(x, 3)', '4x4 矩阵'),
               'cross': ('torch.linalg.cross(x, y)', '4 个三维向量与常量 y 的叉积'), 'matmul': ('torch.linalg.matmul(x, y)', 'y 为 4x4 常量'), 'vecdot': ('torch.linalg.vecdot(x, y)', '逐行点积，y 为 4x4 常量'),
               'multi_dot': ('torch.linalg.multi_dot([x, y, z])', 'y 4x3、z 3x4 常量'), 'householder_product': ('torch.linalg.householder_product(x, tau)', '6x3 的反射向量与常量 tau，输出 H 的前 3 列'),
               'tensorinv': ('torch.linalg.tensorinv(x, ind=1)', '输入 4x2x2，输出 2x2x4'), 'tensorsolve': ('torch.linalg.tensorsolve(x, B)', '输入 2x2x4，B 为 2x2 常量，输出 4'),
               'vander': ('torch.linalg.vander(x)', '长 4 的向量，输出 4x4 范德蒙德矩阵'),
               'cholesky_ex': ('torch.linalg.cholesky_ex(x)[0]', 'info = 0（导出时断言），比对 L'), 'inv_ex': ('torch.linalg.inv_ex(x)[0]', 'info = 0，比对逆'),
               'solve_ex': ('torch.linalg.solve_ex(x, B)[0]', 'info = 0，B 为 4x2 常量'), 'lu_factor_ex': ('torch.linalg.lu_factor_ex(x)[0]', 'info = 0，比对紧凑 LU'),
               'ldl_factor': ('torch.linalg.ldl_factor(x)[0]', '对称正定输入，pivots 为单位置换，比对 LD 的下三角'), 'ldl_factor_ex': ('torch.linalg.ldl_factor_ex(x)[0]', 'info = 0，比对 LD 的下三角'),
               'ldl_solve': ('torch.linalg.ldl_solve(LD, pivots, x)', '输入为右端项 B（4x2），LD 与 pivots 是常量')}
    calls.update(lacalls)
    lanote = {'inv': 'Newton–Schulz 迭代 X <- X (2I - A X)，X0 = Aᵀ / (‖A‖₁‖A‖∞)，30 次，每次两个 4x4 矩阵乘',
              'lu': '无主元的 Gauss 消元（三个 Gauss 变换 I - l_k e_kᵀ，掩码取列 k 主元下方）；输入对角占优，LAPACK 也不选主元（导出时断言 pivots 为单位置换）',
              'tri': '三角矩阵求逆用幂零级数精确展开：(I + N)⁻¹ = I - N + N² - N³（N 严格三角，N⁴ = 0）；上三角 U = D (I + M)',
              'jacobi': '循环 Jacobi 旋转（10 遍 x 6 个 (p, q) 对，旋转角用稳定公式 t = 2a_pq·sgn(d) / (|d| + √(d² + 4a_pq²))），特征值用"比它小的个数"做名次的置换矩阵排序，特征向量符号按第一分量归一（参考同样归一）',
              'svd': 'Jacobi 求 AᵀA 的特征分解得 V 与 σ = √w（降序），U = A V / σ；参考 svd 的 V 列符号同样按第一分量归一、U 随之',
              'cat': '多个输出用矩阵乘拼接：[A | B] = A @ [I 0] + B @ [0 I]（常量 buffer），不用 tensor.concat'}
    def la(*keys, extra=''): return '`torch.linalg` 的分解与求解在 torch-mlir 中没有 lowering（LAPACK 类算子）。导出图是定长的 linalg 组合：' + '；'.join(lanote[k] for k in keys) + ('；' + extra if extra else '') + '。'
    lare = {'norm': '`aten.linalg_norm` 的 lowering 失败；导出图用 √Σx²。', 'vector_norm': '导出的图是基于 pow 的归约，hwacha-mlir 不生成内核（no kernels found）；导出图改用 √Σx²。', 'matrix_norm': '导出的图是基于 pow 的归约，hwacha-mlir 不生成内核（no kernels found）；导出图改用 √Σx²。', 'det': la('lu', extra='det = Π diag(U)'), 'slogdet': la('lu', extra='sign = Π sgn(u_ii)、logabsdet = Σ log|u_ii|，两个标量经 one-hot 拼成长 2 的向量'),
            'cond': la('jacobi', 'svd', extra='cond = σ₀ / σ₃'), 'matrix_rank': la('jacobi', 'svd', extra='rank = #{σ > rtol·σmax}，秩 3 的 6x4 矩阵'),
            'cholesky': la('lu', extra='对称正定矩阵 A = L D Lᵀ，D = diag(U)，Cholesky 因子 = L·√D'), 'cholesky_ex': la('lu', extra='同 cholesky，info 恒为 0'),
            'qr': '改进 Gram–Schmidt（逐列投影、归一，Q = Σ q_j e_jᵀ，R = QᵀA）。' + lanote['cat'] + '。LAPACK 的 Householder QR 允许 R 对角为负，参考按 sgn(diag R) 把 Q 的列、R 的行归一到 diag(R) > 0。',
            'lu': la('lu', 'cat'), 'lu_factor': la('lu', extra='输出 L - I + U'), 'lu_factor_ex': la('lu', extra='输出 L - I + U，info 恒为 0'),
            'eigh': la('jacobi', 'cat'), 'eigvalsh': la('jacobi'), 'eig': la('jacobi', 'cat', extra='输入对称，参考的复特征值取实部按升序排列、特征向量取实部并归一符号'), 'eigvals': la('jacobi', extra='参考取实部升序'),
            'svd': la('jacobi', 'svd', 'cat'), 'svdvals': la('jacobi', 'svd'),
            'solve': la('inv', extra='X = A⁻¹B'), 'solve_ex': la('inv', extra='X = A⁻¹B，info 恒为 0'), 'inv': la('inv'), 'inv_ex': la('inv', extra='info 恒为 0'),
            'solve_triangular': la('tri'), 'lu_solve': la('tri', extra='X = U⁻¹ L⁻¹ B，L、U 取自常量 LU'), 'lstsq': la('inv', extra='满列秩：解 = (AᵀA)⁻¹AᵀB'), 'pinv': la('inv', extra='满列秩：pinv = (AᵀA)⁻¹Aᵀ'),
            'matrix_exp': '`aten.linalg_matrix_exp` 没有 lowering。导出图用缩放平方：exp(A) = (exp(A/4))⁴，exp(A/4) 用 Horner 法则算 15 项 Taylor。',
            'cross': '`aten.linalg_cross` 的 lowering 要求最后一维为 3 且形状匹配；导出图用常量下标的 `index_select` 写出叉积公式 x[i1]·y[i2] - x[i2]·y[i1]。',
            'householder_product': '`aten.linalg_householder_product` 没有 lowering。导出图显式相乘 H = Π (I - τ_i v_i v_iᵀ)，v_i 由输入第 i 列取 i 以下的分量并令 v_i[i] = 1。',
            'tensorinv': la('inv', extra='先 reshape 成 4x4 求逆再 reshape 回 2x2x4'), 'tensorsolve': la('inv', extra='reshape 成 4x4 后 x = A⁻¹b'),
            'vander': '`torch.vander` 会 lower 成 `tm_tensor.scan`（cumprod）；导出图用重复相乘得到 x⁰..x³，再经 one-hot 外积拼成矩阵。',
            'ldl_factor': la('lu', extra='对称正定矩阵 A = L D Lᵀ：LD = (L - I) + diag(U)，sytrf 对此输入不选主元（导出时断言）'), 'ldl_factor_ex': la('lu', extra='同 ldl_factor，info 恒为 0'),
            'ldl_solve': la('tri', extra='X = L⁻ᵀ D⁻¹ L⁻¹ B，L、D 取自常量 LD')}
    rewritten.update(lare)
    spcalls = {'entr': ('torch.special.entr(x)', 'x ∈ [0.5, 3]'), 'erf': ('torch.special.erf(x)', 'x ∈ [-2, 2]'), 'expit': ('torch.special.expit(x)', 'x ∈ [-2, 2]'),
               'expm1': ('torch.special.expm1(x)', 'x ∈ [-2, 2]'), 'log1p': ('torch.special.log1p(x)', 'x ∈ [0.5, 3]'), 'logit': ('torch.special.logit(x)', 'x ∈ [0.05, 0.95]'),
               'log_softmax': ('torch.special.log_softmax(x, 1)', '4 x 16，沿最后一维'), 'softmax': ('torch.special.softmax(x, 1)', '4 x 16，沿最后一维'), 'logsumexp': ('torch.special.logsumexp(x, 1)', '4 x 16，沿最后一维'),
               'ndtr': ('torch.special.ndtr(x)', 'x ∈ [-2, 2]'), 'round': ('torch.special.round(x)', 'x ∈ [-6, 6]'), 'sinc': ('torch.special.sinc(x)', 'x ∈ [-2, 2]'),
               'xlogy': ('torch.special.xlogy(x, y)', 'y 为常量，x, y ∈ [0.5, 3]'), 'xlog1py': ('torch.special.xlog1py(x, y)', 'y 为常量，x, y ∈ [0.5, 3]'), 'exp2': ('torch.special.exp2(x)', 'x ∈ [-2, 2]'),
               'erfc': ('torch.special.erfc(x)', 'x ∈ [-2, 2]'), 'erfcx': ('torch.special.erfcx(x)', 'x ∈ [0, 2]'), 'log_ndtr': ('torch.special.log_ndtr(x)', 'x ∈ [-2, 3]'),
               'erfinv': ('torch.special.erfinv(x)', 'x ∈ [-0.9, 0.9]'), 'ndtri': ('torch.special.ndtri(x)', 'x ∈ [0.05, 0.95]'),
               'gammaln': ('torch.special.gammaln(x)', 'x ∈ [0.5, 4]'), 'digamma': ('torch.special.digamma(x)', 'x ∈ [0.5, 4]'), 'psi': ('torch.special.psi(x)', 'x ∈ [0.5, 4]'),
               'polygamma': ('torch.special.polygamma(1, x)', 'n = 1（trigamma），x ∈ [0.5, 4]'), 'multigammaln': ('torch.special.multigammaln(x, 3)', 'p = 3，x ∈ [2.5, 5.5]'),
               'gammainc': ('torch.special.gammainc(a, x)', 'a = 2.5 常量，x ∈ [0.5, 5]'), 'gammaincc': ('torch.special.gammaincc(a, x)', 'a = 2.5 常量，x ∈ [0.5, 5]'), 'zeta': ('torch.special.zeta(s, x)', 's = 3.5 常量（Hurwitz zeta），x ∈ [1, 5]'),
               'bessel_j0': ('torch.special.bessel_j0(x)', 'x ∈ [0.5, 4]'), 'bessel_j1': ('torch.special.bessel_j1(x)', 'x ∈ [0.5, 4]'), 'bessel_y0': ('torch.special.bessel_y0(x)', 'x ∈ [0.5, 4]'), 'bessel_y1': ('torch.special.bessel_y1(x)', 'x ∈ [0.5, 4]'),
               'modified_bessel_i0': ('torch.special.modified_bessel_i0(x)', 'x ∈ [0.5, 3]'), 'modified_bessel_i1': ('torch.special.modified_bessel_i1(x)', 'x ∈ [0.5, 3]'),
               'modified_bessel_k0': ('torch.special.modified_bessel_k0(x)', 'x ∈ [0.5, 3]'), 'modified_bessel_k1': ('torch.special.modified_bessel_k1(x)', 'x ∈ [0.5, 3]'),
               'i0': ('torch.special.i0(x)', 'x ∈ [0.5, 3]'), 'i1': ('torch.special.i1(x)', 'x ∈ [0.5, 3]'), 'i0e': ('torch.special.i0e(x)', 'x ∈ [0.5, 3]'), 'i1e': ('torch.special.i1e(x)', 'x ∈ [0.5, 3]'),
               'scaled_modified_bessel_k0': ('torch.special.scaled_modified_bessel_k0(x)', 'x ∈ [0.5, 3]'), 'scaled_modified_bessel_k1': ('torch.special.scaled_modified_bessel_k1(x)', 'x ∈ [0.5, 3]'),
               'spherical_bessel_j0': ('torch.special.spherical_bessel_j0(x)', 'x ∈ [0.5, 4]'), 'airy_ai': ('torch.special.airy_ai(x)', 'x ∈ [-2, 2]'),
               'hermite_polynomial_h': ('torch.special.hermite_polynomial_h(x, 5)', 'x ∈ [-1, 1]'), 'hermite_polynomial_he': ('torch.special.hermite_polynomial_he(x, 5)', 'x ∈ [-1, 1]'),
               'laguerre_polynomial_l': ('torch.special.laguerre_polynomial_l(x, 5)', 'x ∈ [0.5, 3]'), 'legendre_polynomial_p': ('torch.special.legendre_polynomial_p(x, 5)', 'x ∈ [-1, 1]')}
    for t in 'tuvw':
        spcalls[f'chebyshev_polynomial_{t}'] = (f'torch.special.chebyshev_polynomial_{t}(x, 5)', 'x ∈ [-1, 1]')
        spcalls[f'shifted_chebyshev_polynomial_{t}'] = (f'torch.special.shifted_chebyshev_polynomial_{t}(x, 5)', 'x ∈ [0, 1]')
    calls.update(spcalls)
    nolow = '`torch.special` 的这个函数在 torch-mlir 中没有 lowering。导出图'
    spre = {'exp2': '导出的 `powf(2, x)` 让 hwacha-mlir 不生成内核；导出图用 exp(x·ln 2)。', 'erfc': nolow + '用 1 - erf(x)（erf 有 lowering；|x| <= 2 无抵消问题）。',
            'erfcx': nolow + '用 exp(x²)(1 - erf x)，x ∈ [0, 2]。', 'log_ndtr': nolow + '用 log(½(1 + erf(x/√2)))。',
            'erfinv': '`aten.erfinv` 的 lowering 失败。导出图用 Giles 的单精度有理逼近（w = -log(1 - x²) 分两段的 9 次多项式）。', 'ndtri': nolow + '用 √2·erfinv(2x - 1)（Giles 逼近）。',
            'gammaln': nolow + '先把自变量平移 8（减去 Σ log(x + k)），再用 Stirling 级数（到 1/z⁷ 项）。', 'digamma': nolow + '先平移 8（减去 Σ 1/(x + k)），再用渐近级数（到 1/z⁸ 项）。',
            'psi': nolow + '同 digamma：平移 8 加渐近级数。', 'polygamma': nolow + '（trigamma）平移 8（加上 Σ 1/(x + k)²），再用渐近级数（到 1/z⁹ 项）。',
            'multigammaln': nolow + '用定义 p(p-1)/4·log π + Σ_j lgamma(x - j/2)，lgamma 用平移 + Stirling。',
            'gammainc': nolow + '用级数 P(a, x) = xᵃe⁻ˣ/Γ(a+1)·Σ_k xᵏ/((a+1)…(a+k))，40 项，Γ(a+1) 为常量。', 'gammaincc': nolow + '用 1 - P(a, x)（同 gammainc 的级数）。',
            'zeta': nolow + '用 Euler–Maclaurin：10 项直接求和 + 尾积分 + 3 项 Bernoulli 修正。',
            'bessel_j0': nolow + '用幂级数 Σ (-1)ᵏ tᵏ/(k!)²（t = x²/4，Horner，25 项）。', 'bessel_j1': nolow + '用幂级数 (x/2) Σ (-1)ᵏ tᵏ/(k!(k+1)!)。',
            'bessel_y0': nolow + '用对数级数 (2/π)[(ln(x/2) + γ) J₀ + Σ (-1)ᵏ⁺¹ H_k tᵏ/(k!)²]。', 'bessel_y1': nolow + '用对数级数（含 J₁、-2/(πx) 与调和数级数）。',
            'modified_bessel_i0': nolow + '用幂级数 Σ tᵏ/(k!)²。', 'modified_bessel_i1': nolow + '用幂级数 (x/2) Σ tᵏ/(k!(k+1)!)。', 'i0': nolow + '用幂级数 Σ tᵏ/(k!)²。', 'i1': nolow + '用幂级数 (x/2) Σ tᵏ/(k!(k+1)!)。',
            'i0e': nolow + '用 e⁻ˣ·I₀（幂级数）。', 'i1e': nolow + '用 e⁻ˣ·I₁（幂级数）。',
            'modified_bessel_k0': nolow + '用对数级数 -(ln(x/2) + γ) I₀ + Σ H_k tᵏ/(k!)²。', 'modified_bessel_k1': nolow + '用对数级数 1/x + (ln(x/2) + γ) I₁ - (x/4) Σ (H_k + H_{k+1}) tᵏ/(k!(k+1)!)。',
            'scaled_modified_bessel_k0': nolow + '用 eˣ·K₀（对数级数）。', 'scaled_modified_bessel_k1': nolow + '用 eˣ·K₁（对数级数）。',
            'spherical_bessel_j0': nolow + '用 sin(x)/x。', 'airy_ai': nolow + '用 Maclaurin 级数 Ai = c₁f - c₂g（20 项）。',
            'hermite_polynomial_h': nolow + '用三项递推 H_{k+1} = 2x H_k - 2k H_{k-1}，n = 5。', 'hermite_polynomial_he': nolow + '用三项递推 He_{k+1} = x He_k - k He_{k-1}，n = 5。',
            'laguerre_polynomial_l': nolow + '用三项递推 (k+1) L_{k+1} = (2k+1-x) L_k - k L_{k-1}，n = 5。', 'legendre_polynomial_p': nolow + '用三项递推 (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}，n = 5。'}
    for t in 'tuvw':
        spre[f'chebyshev_polynomial_{t}'] = nolow + f'用三项递推 P_{{k+1}} = 2x P_k - P_{{k-1}}（P₁ 按 {t.upper()} 的定义），n = 5。'
        spre[f'shifted_chebyshev_polynomial_{t}'] = nolow + f'用 {t.upper()}(2x - 1) 的三项递推，n = 5。'
    rewritten.update(spre)
    rewritten['fftfreq'] = rewritten['rfftfreq'] = '`aten.fft_fftfreq` / `aten.fft_rfftfreq` 在 torch-mlir 中没有 lowering。频率向量本就是常量，导出图把它作为 buffer 加到输入上。'
    rewritten['fftshift'] = rewritten['ifftshift'] = '`torch.roll` 会 lower 成 slice + concat；导出图用常量下标向量的 `index_select` 逐维做同样的循环移位。'
    rewritten['topk'] = '`aten.topk` 在 torch-mlir 中 lower 成 `tm_tensor.sort`，hwacha-mlir 不接受（与 `../torchfunc` 的 fold / max_unpool 同一限制）。导出图用等价的 linalg 组合：每个元素的名次 = 本行中严格大于它的元素个数（随机数据无并列），名次为 i 的元素用 one-hot 求和选出：`values_i = sum_j x_j [rank_j == i]`，`indices_i = sum_j j [rank_j == i]`。每行 O(N^2) 次比较而不是排序，值与下标都精确。'
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        m, x = ef.build(case); m = m.eval()
        with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
        call, note = calls.get(case, (f'torch.{case}(x)', ''))
        t = f'# {case}\n\n'
        qual = f'fft.{case}' if case in rewritten and rewritten[case] is fftnote or case in ('fftshift', 'ifftshift', 'fftfreq', 'rfftfreq') else f'signal.windows.{case}' if case in wincalls else f'linalg.{case}' if case in lacalls else f'special.{case}' if case in spcalls else case
        t += f'`torch.{qual}` 的单函数测试，一次调用，与 PyTorch 逐元素比对。文档：{docs % qual}\n\n'
        t += f'调用：`{call}`' + (f'（{note}）' if note else '') + '\n\n'
        if case in rewritten: t += f'**注意**：本 case 的导出图与参考不是同一段代码。{rewritten[case]} `check.bin` 中的参考值仍由真正的 `torch.{qual}` 算出，导出前脚本断言两者一致。\n\n'
        t += '来源：`export_intf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机数据、固定种子。\n\n'
        t += '## 形状\n\n| | 形状 |\n|---|---|\n' + f'| 输入 `x` | {shp(x)} |\n| 输出 | {shp(y)} |\n'
        b = bufs_of(m)
        if b: t += f'| 常量 buffer | {b} |\n'
        has_lib = os.path.exists(os.path.join(D, case, 'hwlib.s'))
        t += '\n## 文件\n\n' + files_table(case, f'{case}.s', [('`mod_main.c`', '通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\\|ref\\|），打印 PASS/FAIL'), (f'`{case}_check.bin`', '输入与 PyTorch 参考输出（`.incbin` 嵌入）'), ('`HWMLIRFLAGS`', 'hwacha-mlir 的映射选项')] + ([('`hwlib.s`', '库内核')] if has_lib else []))
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 PyTorch 重新生成\n```\n\n' + f'hwacha-mlir 映射：`{flags(case) or "默认"}`\n\n'
        t += '## Spike 结果\n\n' + fmt_res(R.get(case)) + '\n'
        write(case, t)

elif d == 'ttmodule':
    ef = load(os.path.join(D, 'export_ttf.py'), 'ef')
    docs = 'https://meta-pytorch.org/torchtune/0.6/generated/torchtune.%s.html'
    info = {   # case -> (qualified name, what the case does, note on the export)
        'multi_head_attention': ('modules.MultiHeadAttention', 'GQA 自注意力：embed 32、4 个 query 头 / 2 个 kv 头、head_dim 8、RoPE、因果掩码；x = y', ''),
        'feed_forward': ('modules.FeedForward', 'SwiGLU 前馈：gate / up 32 -> 64，down 64 -> 32', ''),
        'kv_cache': ('modules.KVCache', '容量 8 的 kv 缓存已写入 4 个位置，再 update() 2 个位置（输入为堆叠的 k、v），输出整个缓存 [k; v]', '缓存的 index-copy 写入没有 lowering；导出图用选择矩阵的矩阵乘把新行加到常量缓存上：base + P @ x'),
        'rotary_positional_embeddings': ('modules.RotaryPositionalEmbeddings', 'head_dim 8、序列 8、4 个头的旋转位置编码', 'RoPE 的 cos/sin 表是非持久 buffer，导出前重新注册为持久 buffer（torch-mlir 导入需要）'),
        'rmsnorm': ('modules.RMSNorm', 'dim 32 的 RMSNorm', ''), 'fp32_layer_norm': ('modules.Fp32LayerNorm', 'dim 32 的 LayerNorm（内部按 fp32 计算）', ''),
        'tanh_gate': ('modules.TanhGate', 'x · tanh(scale)，scale 置为 0.5（默认 0 会输出全零）', ''),
        'tied_linear': ('modules.TiedLinear', '与 Embedding(16, 32) 的权重绑定的线性层：32 -> 16', ''),
        'transformer_self_attention_layer': ('modules.TransformerSelfAttentionLayer', 'pre-norm 自注意力层：RMSNorm -> GQA 注意力 -> 残差 -> RMSNorm -> SwiGLU -> 残差', ''),
        'transformer_cross_attention_layer': ('modules.TransformerCrossAttentionLayer', '交叉注意力层：query 来自 8 个 token，key/value 来自 6 个常量 encoder 向量', ''),
        'transformer_decoder': ('modules.TransformerDecoder', '两层解码器：Embedding(16, 32) -> 2 x 自注意力层 -> RMSNorm -> 输出 32 -> 16 logits；输入 8 个 token id（float 转 long）', ''),
        'vision_transformer': ('modules.VisionTransformer', '1 张 8x8 图、patch 4 -> 4 个 patch + CLS，1 层 transformer，输出 token 序列 (1, 1, 1, 5, 32)', ''),
        'layer_dropout': ('modules.LayerDropout', 'prob 0.5 的 LayerDropout 包住一个 FeedForward，eval 模式下被包函数无条件执行', ''),
        'prepare_layer_dropout': ('modules.prepare_layer_dropout', '对两个 FeedForward 调用 prepare_layer_dropout(prob_max=0.2) 后顺序执行（eval：恒等包装）', ''),
        'ce_with_chunked_output_loss': ('modules.loss.CEWithChunkedOutputLoss', '2 个 chunk 的交叉熵：logits (1, 8, 16) 切成两半，标签含一个 ignore_index', ''),
        'forward_kl_loss': ('modules.loss.ForwardKLLoss', '学生 / 教师 logits (1, 8, 16) 的前向 KL，标签含一个 ignore_index', '损失对未掩码 token 数做数据相关分支（if sum_masks == 0），torch.export 无法追踪；导出图写出公式 -Σ_i m_i Σ_v p_t log p_s / Σ_i m_i'),
        'forward_kl_with_chunked_output_loss': ('modules.loss.ForwardKLWithChunkedOutputLoss', '2 个 chunk 的前向 KL', '同 forward_kl_loss：导出图写出公式（chunk 求和等价于整体）'),
        'lora_linear': ('modules.peft.LoRALinear', '32 -> 16、rank 4、alpha 8 的 LoRA 线性层（lora_b 随机初始化，默认为零）', ''),
        'dora_linear': ('modules.peft.DoRALinear', '32 -> 16、rank 4、alpha 8 的 DoRA 线性层，magnitude 由 initialize_dora_magnitude 计算', ''),
        'adapter_module': ('modules.peft.AdapterModule', '实现 AdapterModule 协议的最小模块：y = base(x) + scale · adapter(x)，adapter_params 返回 adapter 权重与 scale', ''),
        'get_adapter_params': ('modules.peft.get_adapter_params', '对 LoRALinear 调用 get_adapter_params（断言得到 lora_a / lora_b），再前向', ''),
        'set_trainable_params': ('modules.peft.set_trainable_params', '只把 adapter 参数设为可训练（断言 base 权重冻结），再前向', ''),
        'get_adapter_state_dict': ('modules.peft.get_adapter_state_dict', '从 LoRALinear 的 state_dict 取出 adapter 权重，用它们重建 LoRA 增量：base(x) + (alpha/rank) x Aᵀ Bᵀ', '导出图用取出的 adapter state dict 重建前向，参考是原 LoRALinear'),
        'validate_missing_and_unexpected_for_lora': ('modules.peft.validate_missing_and_unexpected_for_lora', '按 q_proj 的 LoRA 配置校验 missing / unexpected 键（通过），再前向 LoRALinear', ''),
        'disable_adapter': ('modules.peft.disable_adapter', '在 disable_adapter 上下文中前向 LoRALinear：只剩 base 线性层', ''),
        'fusion_layer': ('modules.model_fusion.FusionLayer', '自注意力层 + 交叉注意力融合层（fusion_first），encoder_input 为 6 个常量向量', ''),
        'fusion_embedding': ('modules.model_fusion.FusionEmbedding', '词表 16 + 融合词表 4 的嵌入，8 个 token 中 4 个落在融合词表', 'masked_select / masked_scatter 没有 lowering；导出图用 one-hot 从拼接的 [E; E_fusion] 表中取行'),
        'deep_fusion_model': ('modules.model_fusion.DeepFusionModel', '解码器（自注意力层 + FusionLayer）+ 线性 encoder，encoder_input 6 个向量，输出 8 个 token 的 logits', ''),
        'register_fusion_module': ('modules.model_fusion.register_fusion_module', '把交叉注意力层注册为融合模块（断言 fusion_params 为其全部参数），再前向', ''),
        'get_fusion_params': ('modules.model_fusion.get_fusion_params', '对 DeepFusionModel 调用 get_fusion_params（断言只含 fusion_layer 的参数），再前向', ''),
        'local_kv_cache': ('modules.common_utils.local_kv_cache', '在 local_kv_cache 上下文中对 8 个 token 做整段 prefill（因果掩码、input_pos 0..7）', '带缓存的前向把 k / v index-copy 进缓存（没有 lowering）；导出图跑同一个解码器的无缓存前向，整段 prefill 下二者相同'),
        'disable_kv_cache': ('modules.common_utils.disable_kv_cache', '已 setup_caches 的解码器在 disable_kv_cache 上下文中前向（不用缓存）', ''),
        'delete_kv_caches': ('modules.common_utils.delete_kv_caches', 'setup_caches 后 delete_kv_caches（断言缓存已删除），再前向', ''),
        'vision_cross_attention_mask': ('modules.transforms.VisionCrossAttentionMask', '由 token 列表 [image_token, 7 个文本 token] 与 1 张 8x8 图（patch 4：4 个 patch + CLS）生成文本 -> 图像掩码，作为交叉注意力层的 encoder_mask', '')}
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        qual, what, note = info.get(case, (case, '', ''))
        m, x = ef.build(case); m = m.eval()
        with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
        t = f'# {case}\n\n'
        t += f'torchtune 的 `torchtune.{qual}` 在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：{docs % qual}\n\n'
        t += f'用例：{what}。\n\n'
        if note: t += f'**注意**：{note}。参考值由真正的 torchtune 调用算出，导出前脚本断言两者一致。\n\n'
        t += '来源：`export_ttf.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子；规模很小（embed 32、4 头 x head_dim 8、序列 8、词表 16）。\n\n'
        t += '## 形状\n\n| | 形状 |\n|---|---|\n' + f'| 输入 `x` | {shp(x)} |\n| 输出 | {shp(y)} |\n'
        has_lib = os.path.exists(os.path.join(D, case, 'hwlib.s'))
        t += '\n## 文件\n\n' + files_table(case, f'{case}.s', [('`mod_main.c`', '通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\\|ref\\|），打印 PASS/FAIL'), (f'`{case}_check.bin`', '输入与 PyTorch 参考输出（`.incbin` 嵌入）'), ('`HWMLIRFLAGS`', 'hwacha-mlir 的映射选项')] + ([('`hwlib.s`', '库内核')] if has_lib else []))
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 PyTorch 重新生成\n```\n\n' + f'hwacha-mlir 映射：`{flags(case) or "默认"}`\n\n'
        t += '## Spike 结果\n\n' + fmt_res(R.get(case)) + '\n'
        write(case, t)

elif d == 'ttmodel':
    ef = load(os.path.join(D, 'export_ttm.py'), 'ef')
    docs = 'https://meta-pytorch.org/torchtune/0.6/generated/torchtune.models.%s.html'
    tiny = 'vocab 16、2 层、4 个 query 头 / 2 个 kv 头（head_dim 8）、embed 32、intermediate 64、序列 8'
    lora_note = 'LoRA 加在 q_proj / v_proj 和 MLP 上（rank 4、alpha 8），lora_b 随机初始化（torchtune 默认置零，adapter 是恒等）'
    info = {   # case -> (doc entry, what it builds, note)
        'llama2': ('llama2.llama2', f'Llama 2 解码器（llama2_7b/13b/70b 的组件构建函数），{tiny}', ''),
        'lora_llama2': ('llama2.lora_llama2', f'LoRA Llama 2 解码器，{tiny}；{lora_note}', ''),
        'llama2_reward': ('llama2.llama2_reward_7b', f'llama2_classifier(num_classes=1)：llama2_reward_7b 所用的奖励模型架构（输出每个 token 一个分数），{tiny}', ''),
        'lora_llama2_reward': ('llama2.lora_llama2_reward_7b', f'lora_llama2_classifier(num_classes=1)，{tiny}；{lora_note}', ''),
        'code_llama2': ('code_llama2.code_llama2_7b', f'Code Llama：code_llama2_7b 所用的 llama2 架构（rope_base 1e6），{tiny}', ''),
        'lora_code_llama2': ('code_llama2.lora_code_llama2_7b', f'LoRA Code Llama（lora_llama2 架构），{tiny}；{lora_note}', ''),
        'llama3': ('llama3.llama3', f'Llama 3 解码器（rope_base 500000、GQA），{tiny}', ''), 'lora_llama3': ('llama3.lora_llama3', f'LoRA Llama 3 解码器，{tiny}；{lora_note}', ''),
        'llama3_1': ('llama3_1.llama3_1', f'Llama 3.1 解码器（scaled RoPE，scale_factor 8），{tiny}', ''), 'lora_llama3_1': ('llama3_1.lora_llama3_1', f'LoRA Llama 3.1 解码器，{tiny}；{lora_note}', ''),
        'llama3_2': ('llama3_2.llama3_2_1b', f'Llama 3.2 解码器（llama3_2_1b/3b 的组件构建函数：tied embeddings、scale_factor 32），{tiny}', ''),
        'lora_llama3_2': ('llama3_2.lora_llama3_2_1b', f'LoRA Llama 3.2 解码器，{tiny}；{lora_note}', ''),
        'llama3_3': ('llama3_3.llama3_3_70b', f'Llama 3.3：llama3_3_70b 所用的 llama3_1 架构，{tiny}', ''), 'lora_llama3_3': ('llama3_3.lora_llama3_3_70b', f'LoRA Llama 3.3（lora_llama3_1 架构），{tiny}；{lora_note}', ''),
        'llama3_2_vision_encoder': ('llama3_2_vision.llama3_2_vision_encoder', 'Llama 3.2 Vision 编码器：CLIP（tile 8、patch 4：4 个 patch + CLS，2 层，embed 32，隐藏状态 [0]）+ 1 层投影头 -> 5 个 token', '单 tile（max_num_tiles=1）：多 tile 的平铺位置编码按每张图的 aspect ratio 循环，torch.export 无法追踪'),
        'lora_llama3_2_vision_encoder': ('llama3_2_vision.lora_llama3_2_vision_encoder', f'LoRA Llama 3.2 Vision 编码器（encoder_lora、fusion_lora），{lora_note}', '单 tile，同 llama3_2_vision_encoder'),
        'llama3_2_vision_decoder': ('llama3_2_vision.llama3_2_vision_decoder', f'Llama 3.2 Vision 融合解码器：每 2 层一个交叉注意力 FusionLayer，FusionEmbedding（2 个特殊 token），encoder_input 为 5 个常量向量，{tiny}', 'FusionEmbedding 的 masked_select / masked_scatter 没有 lowering；导出的副本把它换成 one-hot 从拼接表 [E; E_fusion] 取行的等价模块'),
        'lora_llama3_2_vision_decoder': ('llama3_2_vision.lora_llama3_2_vision_decoder', f'LoRA Llama 3.2 Vision 融合解码器（decoder_lora、fusion_lora），{lora_note}', '同 llama3_2_vision_decoder：FusionEmbedding 换成 one-hot 等价模块'),
        'llama3_2_vision': ('llama3_2_vision.llama3_2_vision_11b', 'llama3_2_vision_11b 的结构：DeepFusionModel(编码器, 融合解码器)，输入 8 个 token 与 1 张 8x8 图，输出 logits', '单 tile 编码器；FusionEmbedding 换成 one-hot 等价模块'),
        'llama3_vision_encoder': ('llama3_2_vision.Llama3VisionEncoder', 'Llama3VisionEncoder 类：clip_vision_encoder（out_indices [0]）+ Llama3VisionProjectionHead', '单 tile'),
        'llama3_vision_projection_head': ('llama3_2_vision.Llama3VisionProjectionHead', '投影头单独运行：1 层 transformer + 输出投影，输入 CLIP 输出 (1,1,1,5,32) 与 1 个隐藏状态', ''),
        'qwen2': ('qwen2.qwen2', f'Qwen2 解码器（带偏置的 q/k/v 投影、rope_base 1e6），{tiny}', ''), 'lora_qwen2': ('qwen2.lora_qwen2', f'LoRA Qwen2 解码器，{tiny}；{lora_note}', ''),
        'qwen2_5': ('qwen2_5.qwen2_5_0_5b', f'Qwen2.5：qwen2_5_0_5b 所用的 qwen2 架构（tie_word_embeddings、norm_eps 1e-6、rope_base 1e6），{tiny}', ''),
        'lora_qwen2_5': ('qwen2_5.lora_qwen2_5_0_5b', f'LoRA Qwen2.5（lora_qwen2 架构，tied embeddings），{tiny}；{lora_note}', ''),
        'phi3': ('phi3.phi3', f'Phi-3 解码器，{tiny}', ''), 'lora_phi3': ('phi3.lora_phi3', f'LoRA Phi-3 解码器，{tiny}；{lora_note}', ''),
        'phi4': ('phi4.phi4_14b', f'Phi-4：phi4_14b 所用的 phi3 架构（rope_base 250000），{tiny}', ''), 'lora_phi4': ('phi4.lora_phi4_14b', f'LoRA Phi-4（lora_phi3 架构），{tiny}；{lora_note}', ''),
        'mistral': ('mistral.mistral', f'Mistral 解码器，{tiny}', ''), 'lora_mistral': ('mistral.lora_mistral', f'LoRA Mistral 解码器，{tiny}；{lora_note}', ''),
        'mistral_reward': ('mistral.mistral_classifier', f'mistral_classifier(num_classes=1)：mistral_reward_7b 所用的奖励模型架构，{tiny}', ''),
        'lora_mistral_reward': ('mistral.lora_mistral_classifier', f'lora_mistral_classifier(num_classes=1)，{tiny}；{lora_note}', ''),
        'gemma': ('gemma.gemma', f'Gemma 解码器（GeLU MLP、tied 输出、embed 缩放），{tiny}', ''), 'lora_gemma': ('gemma.lora_gemma', f'LoRA Gemma 解码器，{tiny}；{lora_note}', ''),
        'gemma2': ('gemma2.gemma2', f'Gemma 2 解码器（logit 与注意力 soft-capping、隔层滑动窗口注意力（窗口 4）、query_pre_attn_scalar），{tiny}', ''),
        'lora_gemma2': ('gemma2.lora_gemma2', f'LoRA Gemma 2 解码器，{tiny}；{lora_note}', ''),
        'clip_vision_encoder': ('clip.clip_vision_encoder', 'CLIP 视觉编码器：tile 8、patch 4（4 个 patch + CLS）、2 层、embed 32，输出 token 序列', '单 tile（max_num_tiles=1）：平铺位置编码按 aspect ratio 循环，torch.export 无法追踪'),
        'token_positional_embedding': ('clip.TokenPositionalEmbedding', '单 tile 的 token 位置编码：x + 可学习的 (5, 32) 表', ''),
        'tiled_token_positional_embedding': ('clip.TiledTokenPositionalEmbedding', '平铺 token 位置编码：2 个 tile（aspect ratio (2, 1)），局部表 + 门控的全局表', '前向按每张图的 aspect ratio 循环并原地累加（数据相关）；固定 aspect ratio (2, 1) 下导出图加上同样的常量切片 local·(1 - tanh g) + global[:2, :1]·tanh g'),
        'tile_positional_embedding': ('clip.TilePositionalEmbedding', 'tile 位置编码：2 个 tile（aspect ratio (2, 1)），每个 tile 一个门控向量', '同 tiled_token_positional_embedding：固定 aspect ratio 下导出图加上常量切片 embedding[:2, :1]·tanh g')}
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        doc, what, note = info.get(case, (case, '', ''))
        m, x = ef.build(case); m = m.eval()
        with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
        t = f'# {case}\n\n'
        t += f'torchtune 模型参考页的 `torchtune.models.{doc}` 所用架构在 Hwacha 上的一次前向，与 PyTorch 逐元素比对。文档：{docs % doc}\n\n'
        t += f'用例：{what}。\n\n'
        if note: t += f'**注意**：{note}。参考值由真正的 torchtune 前向算出，导出前脚本断言两者一致。\n\n'
        t += '来源：`export_ttm.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机权重与输入、固定种子。带尺寸的构建函数（7b、0.5b …）是同一组件构建函数加固定超参数，远超 Spike 的规模，这里用微型尺寸实例化同一架构。\n\n'
        t += '## 形状\n\n| | 形状 |\n|---|---|\n' + f'| 输入 `x` | {shp(x)} |\n| 输出 | {shp(y)} |\n'
        has_lib = os.path.exists(os.path.join(D, case, 'hwlib.s'))
        t += '\n## 文件\n\n' + files_table(case, f'{case}.s', [('`mod_main.c`', '通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\\|ref\\|），打印 PASS/FAIL'), (f'`{case}_check.bin`', '输入与 PyTorch 参考输出（`.incbin` 嵌入）'), ('`HWMLIRFLAGS`', 'hwacha-mlir 的映射选项')] + ([('`hwlib.s`', '库内核')] if has_lib else []))
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 PyTorch 重新生成\n```\n\n' + f'hwacha-mlir 映射：`{flags(case) or "默认"}`\n\n'
        t += '## Spike 结果\n\n' + fmt_res(R.get(case)) + '\n'
        write(case, t)

elif d == 'tvintf':
    ef = load(os.path.join(D, 'export_tvi.py'), 'ef')
    docs = 'https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.%s.html'
    pinned = '随机变换在 forward 里抽参数、掷是否应用的硬币（即使 p=1、范围退化，对 torch.export 也是数据相关），因此固定为该类实际调用的 functional 算子和一次抽样'
    info = {   # case -> (class, what the case does, note)
        'resize': ('Resize', 'Resize(8)：最短边到 8（16x16 -> 8x8），双线性、无抗锯齿', ''), 'resize_bilinear': ('Resize', 'Resize((24, 24), BILINEAR)：放大到 24x24', ''),
        'center_crop': ('CenterCrop', 'CenterCrop(8)', ''), 'random_crop': ('RandomCrop', 'RandomCrop(8) 的一次抽样：F.crop(x, top=3, left=5, 8, 8)', pinned),
        'random_resized_crop': ('RandomResizedCrop', 'RandomResizedCrop(16) 的一次抽样：F.resized_crop(2, 3, 10, 12 -> 16x16, BILINEAR)', pinned),
        'random_horizontal_flip': ('RandomHorizontalFlip', 'F.horizontal_flip', pinned), 'random_vertical_flip': ('RandomVerticalFlip', 'F.vertical_flip', pinned),
        'pad': ('Pad', 'Pad(2)：常数 0 填充到 20x20', ''), 'pad_reflect': ('Pad', 'Pad(3, padding_mode="reflect")：反射填充到 22x22', ''),
        'random_zoom_out': ('RandomZoomOut', 'RandomZoomOut 的一次抽样：F.pad(x, [4, 2, 3, 5], fill=0.5)', pinned),
        'random_rotation': ('RandomRotation', 'RandomRotation(30) 的一次抽样：F.rotate(x, 30°, BILINEAR)', pinned), 'random_rotation_nearest': ('RandomRotation', 'F.rotate(x, 45°, NEAREST)', pinned),
        'random_affine': ('RandomAffine', 'RandomAffine 的一次抽样：F.affine(angle 15°, scale 1.2, shear 10°, BILINEAR)', pinned),
        'random_perspective': ('RandomPerspective', 'RandomPerspective 的一次抽样：四角 [(0,0),(15,0),(0,15),(15,15)] -> [(1,2),(13,1),(2,14),(14,12)]，BILINEAR', pinned + '。8 个透视系数的最小二乘（aten.linalg_lstsq 没有 lowering）在 host 上解出，以 coefficients 传入'),
        'elastic_transform': ('ElasticTransform', 'ElasticTransform 的一次位移场抽样（1x16x16x2，σ 0.05），BILINEAR', pinned),
        'five_crop': ('FiveCrop', 'FiveCrop(8)：四角 + 中心，堆叠为 5x3x8x8', ''), 'ten_crop': ('TenCrop', 'TenCrop(8)：FiveCrop 及其水平翻转，10x3x8x8', ''),
        'random_iou_crop': ('RandomIoUCrop', 'RandomIoUCrop 一次被接受的抽样对应的裁剪：F.crop(4, 2, 10, 12)（框的筛选不导出）', pinned),
        'random_resize': ('RandomResize', 'RandomResize(12, 12)：F.resize 到 12x12', pinned), 'random_shortest_size': ('RandomShortestSize', 'RandomShortestSize(12) 在 24x24 图上的一次抽样：F.resize 到 12x12', pinned),
        'scale_jitter': ('ScaleJitter', 'ScaleJitter 的一次抽样（比例 1.25）：F.resize 到 20x20', pinned),
        'color_jitter': ('ColorJitter', 'ColorJitter 的一次抽样按类的顺序：brightness 1.3 -> contrast 0.8 -> saturation 1.4 -> hue 0.1', pinned + '。F.adjust_hue 的 RGB<->HSV 在 torch-mlir 的 lowering 中失败（arith.cmpi 操作数类型不一致）；导出图用无原地操作、无 aminmax 的同一公式（select 用 one-hot 而不是 gather）重写 rgb_to_hsv / hsv_to_rgb'),
        'grayscale': ('Grayscale', 'Grayscale()：1 通道', ''), 'grayscale_3ch': ('Grayscale', 'Grayscale(num_output_channels=3)', ''), 'random_grayscale': ('RandomGrayscale', 'F.rgb_to_grayscale(x, 3)', pinned),
        'rgb': ('RGB', 'RGB()：单通道图扩成 3 通道', ''), 'random_channel_permutation': ('RandomChannelPermutation', 'F.permute_channels(x, [2, 0, 1]) 的一次抽样', pinned),
        'random_photometric_distort': ('RandomPhotometricDistort', 'RandomPhotometricDistort 的一次抽样：brightness 0.9 -> contrast 1.3 -> saturation 0.7 -> hue -0.05', pinned + '。adjust_hue 同 color_jitter 的重写'),
        'random_adjust_sharpness': ('RandomAdjustSharpness', 'F.adjust_sharpness(x, 2.0)', pinned), 'random_autocontrast': ('RandomAutocontrast', 'F.autocontrast', pinned + '。F.autocontrast 的导出含 tm_tensor.scan / scatter（索引赋值）；导出图用同一公式 (x - min) / (max - min)、torch.where 处理 max == min'),
        'random_equalize': ('RandomEqualize', 'F.equalize 作用于 uint8 图（x·255），结果除以 255', pinned + '。equalize 的 scatter_add 直方图导出时崩溃；导出图按 torchvision 的算法重写：one-hot 计数得 256 bin 直方图、上三角矩阵乘得累计直方图、step 规则得 LUT、one-hot 查表；整数除法用 float32 上的精确 floor'),
        'random_invert': ('RandomInvert', 'F.invert', pinned), 'random_posterize': ('RandomPosterize', 'F.posterize(x·255 as uint8, bits=3) / 255', pinned), 'random_solarize': ('RandomSolarize', 'F.solarize(x, 0.5)', pinned),
        'gaussian_blur': ('GaussianBlur', 'GaussianBlur(5, sigma=1.5)：F.gaussian_blur', 'GaussianBlur 在 forward 里抽 sigma（数据相关）；固定为 F.gaussian_blur(kernel 5, sigma 1.5)'),
        'gaussian_noise': ('GaussianNoise', 'GaussianNoise(sigma=0.1) 的一次噪声抽样：x + 0.1·n，n 为常量', pinned), 'random_erasing': ('RandomErasing', 'RandomErasing 的一次框抽样：F.erase(3, 4, 6, 7, v=0)', pinned),
        'normalize': ('Normalize', 'Normalize(ImageNet mean / std)', ''), 'linear_transformation': ('LinearTransformation', '768x768 随机变换矩阵与均值向量', ''),
        'cutmix': ('CutMix', 'CutMix 的一次框抽样：2 张图，框 [4:12, 2:10] 换成另一张（标签混合不导出）', pinned), 'mixup': ('MixUp', 'MixUp，λ = 0.7：0.7·x + 0.3·x.flip(0)（标签混合不导出）', pinned),
        'uniform_temporal_subsample': ('UniformTemporalSubsample', 'UniformTemporalSubsample(4) 作用于 8 帧视频', '')}
    opsdocs = 'https://docs.pytorch.org/vision/stable/generated/torchvision.ops.%s.html'
    nolow = '这是 torchvision 的 C++ 算子（torch.ops.torchvision.*），torch-mlir 没有 lowering'
    opsinfo = {   # case -> (ops entry, what, note)
        'box_area': ('box_area', '8 个 xyxy 框的面积', ''), 'box_convert': ('box_convert', "box_convert(x, 'xyxy', 'cxcywh')", ''),
        'box_iou': ('box_iou', '8 x 8 的 IoU 矩阵，第二组框为常量', ''), 'generalized_box_iou': ('generalized_box_iou', '8 x 8 的 GIoU 矩阵', ''),
        'distance_box_iou': ('distance_box_iou', '8 x 8 的 DIoU 矩阵', ''), 'complete_box_iou': ('complete_box_iou', '8 x 8 的 CIoU 矩阵', ''),
        'clip_boxes_to_image': ('clip_boxes_to_image', '裁到 32 x 48 的图像范围', ''),
        'remove_small_boxes': ('remove_small_boxes', 'min_size 8，输出保留掩码（0/1 向量）', 'remove_small_boxes 返回下标（长度数据相关）；导出图输出等价的保留掩码，参考把下标转成掩码比对'),
        'masks_to_boxes': ('masks_to_boxes', '3 张 16x16 的 0/1 掩码 -> 3 个 xyxy 框', 'masks_to_boxes 用 nonzero（长度数据相关）；导出图对常量坐标网格按掩码取 min / max'),
        'nms': ('nms', '8 个框、常量分数、IoU 阈值 0.5，输出输入顺序下的保留掩码', nolow + '。导出图：按分数的名次置换排序，贪心抑制在 8 个框上展开（框 i 被保留当且仅当没有更早保留的框与它 IoU > 阈值）；nms 返回按分数排序的保留下标，参考转成同样的掩码'),
        'batched_nms': ('batched_nms', '8 个框分 3 类，类内 NMS，输出保留掩码', nolow + '。导出图：框加上 类别 x 100 的偏移后做同样的贪心 NMS（torchvision 的 offset 技巧）'),
        'sigmoid_focal_loss': ('sigmoid_focal_loss', "4 x 8 logits、0/1 目标，reduction='sum'", ''),
        'generalized_box_iou_loss': ('generalized_box_iou_loss', "8 对框的 GIoU 损失之和", 'torchvision 的 _loss_inter_union 通过布尔掩码赋值，torch-mlir 的 lowering 失败；导出图用 torch.where 写同一公式'),
        'distance_box_iou_loss': ('distance_box_iou_loss', "8 对框的 DIoU 损失之和", '同 generalized_box_iou_loss：torch.where 写同一公式'), 'complete_box_iou_loss': ('complete_box_iou_loss', "8 对框的 CIoU 损失之和", '同 generalized_box_iou_loss：torch.where 写同一公式'),
        'roi_align': ('roi_align', '1x4x16x16 特征图、3 个常量 RoI、输出 2x2、spatial_scale 0.5、sampling_ratio 2、aligned=False', nolow + '（torchvision 的纯张量 _roi_align 同样 lowering 失败）。导出图按 CUDA 内核的采样规则实现：每个 bin 2x2 个双线性采样点求平均，越界一个像素以内的点夹到边界、更远的读 0，用 one-hot 矩阵做 gather'),
        'roi_align_aligned': ('roi_align', '同 roi_align，spatial_scale 1、aligned=True（坐标减 0.5）', '同 roi_align 的实现'), 'roi_align_module': ('RoIAlign', 'RoIAlign(2, 0.5, 2) 模块', '同 roi_align 的实现'),
        'roi_pool': ('roi_pool', '1x4x16x16 特征图、3 个常量 RoI、输出 2x2 的最大池化', nolow + '。RoI 是常量，每个 bin 覆盖的像素集合在 host 上按 torchvision 的取整规则算成 0/1 掩码；导出图对掩码内像素取最大'),
        'roi_pool_module': ('RoIPool', 'RoIPool(2, 1.0) 模块', '同 roi_pool 的实现'),
        'ps_roi_align': ('ps_roi_align', '1x16x16x16 特征图（C·k·k = 4·2·2）、3 个 RoI、输出 2x2、sampling_ratio 2', nolow + '。导出图：位置敏感版本的双线性采样（bin (i, j) 只读通道 c·k·k + i·k + j），采样点零填充'),
        'ps_roi_align_module': ('PSRoIAlign', 'PSRoIAlign(2, 1.0, 2) 模块', '同 ps_roi_align 的实现'),
        'ps_roi_pool': ('ps_roi_pool', '1x16x16x16 特征图、3 个不超过 10 像素的 RoI、输出 2x2', nolow + '。导出图用 `../deformable/dcn_ops.py` 的 ps_roi_pool_ref（每个 bin 读固定 6x6 的候选窗口并掩掉超出部分）'),
        'ps_roi_pool_module': ('PSRoIPool', 'PSRoIPool(2, 1.0) 模块', '同 ps_roi_pool 的实现'),
        'multi_scale_roi_align': ('MultiScaleRoIAlign', "单一层级 ['0']、输出 2、sampling_ratio 2，64x64 图像上推断 scale = 1/4", nolow + '。单层级下等于 roi_align(scale 1/4, aligned=False)，同 roi_align 的实现'),
        'deform_conv2d': ('deform_conv2d', '3x3、padding 1 的可变形卷积（DCN v1）：1x4x16x16 -> 1x6x16x16，偏移为常量', nolow + '。导出图用 `../deformable/dcn_ops.py` 的 deform_conv2d_ref（gather + 双线性 + 矩阵乘）'),
        'deform_conv2d_mask': ('deform_conv2d', '带调制掩码的可变形卷积（DCN v2）', '同 deform_conv2d 的实现'), 'deform_conv2d_module': ('DeformConv2d', 'DeformConv2d(4, 6, 3, padding=1) 模块', '同 deform_conv2d 的实现'),
        'conv2d_norm_activation': ('Conv2dNormActivation', 'Conv2dNormActivation(4, 8, 3)：卷积 + BatchNorm（eval）+ ReLU', ''), 'conv3d_norm_activation': ('Conv3dNormActivation', 'Conv3dNormActivation(2, 4, 3)', ''),
        'frozen_batch_norm2d': ('FrozenBatchNorm2d', 'FrozenBatchNorm2d(4)，随机的仿射参数与统计量', ''), 'mlp': ('MLP', 'MLP(8, [16, 4])', ''), 'permute': ('Permute', 'Permute([0, 2, 3, 1])', ''),
        'squeeze_excitation': ('SqueezeExcitation', 'SqueezeExcitation(4, 2)', ''), 'drop_block2d': ('DropBlock2d', 'DropBlock2d(0.3, 3)，eval 模式：恒等', ''), 'drop_block3d': ('DropBlock3d', 'DropBlock3d(0.3, 3)，eval：恒等', ''),
        'stochastic_depth': ('StochasticDepth', "StochasticDepth(0.3, 'row')，eval：恒等", ''), 'drop_block2d_fn': ('drop_block2d', 'drop_block2d(x, 0.3, 3, training=False)：恒等', ''), 'drop_block3d_fn': ('drop_block3d', 'drop_block3d(x, 0.3, 3, training=False)：恒等', ''),
        'stochastic_depth_fn': ('stochastic_depth', "stochastic_depth(x, 0.3, 'row', training=False)：恒等", ''),
        'feature_pyramid_network': ('FeaturePyramidNetwork', "FeaturePyramidNetwork([4, 4], 4) 作用于两个层级（x 与其 2x2 平均池化），输出两级展平后拼接", '')}
    utdocs = 'https://docs.pytorch.org/vision/stable/generated/torchvision.utils.%s.html'
    utinfo = {'make_grid': ('make_grid', '4 张 3x8x8 图，nrow 2、padding 1、pad_value 0.5 -> 3x19x19 网格', ''),
              'make_grid_normalize': ('make_grid', 'normalize=True、scale_each=True：每张图按自身 min / max 归一后拼网格', 'normalize 把张量自身的 min / max 取成 python 数做 clamp 边界（数据相关）；导出图用张量运算做同样的逐图归一化再 make_grid'),
              'save_image': ('save_image', 'save_image 写入 PNG 的像素：make_grid 后 x·255 + 0.5、clamp、截断为 uint8', '参考值是真正 save_image 写出的 PNG 读回来的像素；导出图算写入前的同一量化'),
              'draw_bounding_boxes': ('draw_bounding_boxes', '16x16 uint8 图上画 3 个 width 2 的彩色框（后画的覆盖先画的）', 'draw_bounding_boxes 用 PIL 光栅化；导出图在坐标网格上按 PIL 的矩形轮廓规则（外框含、内缩 width 的框不含）逐框覆盖，与 PIL 逐像素一致'),
              'draw_segmentation_masks': ('draw_segmentation_masks', '2 个重叠的掩码、alpha 0.6，两种颜色', 'torchvision 通过布尔索引赋值（无 lowering）；导出图按同一规则：每个掩码涂色、重叠像素置 0、再与原图按 alpha 混合并截断'),
              'draw_keypoints': ('draw_keypoints', '3 个关键点、radius 2、红色实心圆', 'draw_keypoints 用 PIL 的 ellipse 光栅化；导出图在坐标网格上填充 d² <= (r + 0.5)² 的像素，与 PIL 在 radius 2（和 3）下逐像素一致（radius 1、4 时 PIL 的光栅化不同，本用例未覆盖）'),
              'flow_to_image': ('flow_to_image', '1x2x8x8 的光流 -> uint8 彩色图', 'flow_to_image 用 atan2 和整数下标查色轮；导出图按同一算法：按最大范数归一、atan2 由 atan 加象限修正得到、色轮的两个相邻条目用 one-hot 查表后线性插值、按范数向白色淡化、截断')}
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        isop = case in opsinfo; isut = case in utinfo
        cls, what, note = opsinfo[case] if isop else utinfo[case] if isut else info.get(case, (case, '', ''))
        m, x = ef.build(case); m = m.eval()
        with torch.no_grad(): y = m.reference(x) if hasattr(m, 'reference') else m(x)
        t = f'# {case}\n\n'
        if isop: t += f'torchvision 的 `torchvision.ops.{cls}` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：{opsdocs % cls}\n\n'
        elif isut: t += f'torchvision 的 `torchvision.utils.{cls}` 在 Hwacha 上的一次调用，与 PyTorch 逐元素比对。文档：{utdocs % cls}\n\n'
        else: t += f'torchvision 的 `torchvision.transforms.v2.{cls}` 在 Hwacha 上的一次应用，与 PyTorch 逐元素比对。文档：{docs % cls}\n\n'
        t += f'用例：{what}。\n\n'
        if note: t += f'**注意**：{note}。参考值由真正的 torchvision 调用算出，导出前脚本断言两者一致。\n\n'
        t += ('来源：`export_tvi.py` / `intf_ops.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。\n\n' if isop else '来源：`export_tvi.py` / `intf_utils.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机输入、固定种子。\n\n' if isut else '来源：`export_tvi.py`（PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc），随机 3x16x16 的 [0, 1] 图像、固定种子。\n\n')
        t += '## 形状\n\n| | 形状 |\n|---|---|\n' + f'| 输入 `x` | {shp(x)} |\n| 输出 | {shp(y)} |\n'
        has_lib = os.path.exists(os.path.join(D, case, 'hwlib.s'))
        t += '\n## 文件\n\n' + files_table(case, f'{case}.s', [('`mod_main.c`', '通用 host：从 check.bin 读入输入，调用 `net(x)`，与参考输出比对（容差 1e-3 + 1e-2·max\\|ref\\|），打印 PASS/FAIL'), (f'`{case}_check.bin`', '输入与 PyTorch 参考输出（`.incbin` 嵌入）'), ('`HWMLIRFLAGS`', 'hwacha-mlir 的映射选项')] + ([('`hwlib.s`', '库内核')] if has_lib else []))
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '.riscv\nmake ' + case + '.spike    # 在 Spike 上运行\nmake gen-' + case + '      # 从 PyTorch 重新生成\n```\n\n' + f'hwacha-mlir 映射：`{flags(case) or "默认"}`\n\n'
        t += '## Spike 结果\n\n' + fmt_res(R.get(case)) + '\n'
        write(case, t)

elif d == 'torchvision':
    import torchvision, torchvision.models as M
    models = {}
    for l in open(os.path.join(D, 'models.txt')):
        if l.startswith('#') or not l.strip(): continue
        n, hw, kw = l.split(None, 2); models[n] = (int(hw), kw.strip())
    params = {}
    pp = os.path.join(D, '.logs', 'params.txt')
    if os.path.exists(pp):
        for l in open(pp): n, v = l.split(); params[n] = float(v)
    def nparams(n):
        try:
            m = M.get_model(n, weights=None, weights_backbone=None) if fam_of(n) in ('segmentation', 'detection') else M.get_model(n, weights=None)
            return sum(p.numel() for p in m.parameters()) / 1e6
        except Exception: return None
    fam_of = lambda n: next((f for f in ('segmentation', 'detection', 'video', 'optical_flow') if n in M.list_models(module=getattr(M, f))), 'classification')
    fam_cn = {'classification': '图像分类', 'segmentation': '语义分割', 'detection': '目标检测', 'video': '视频分类', 'optical_flow': '光流'}
    fam_doc = {'classification': 'https://docs.pytorch.org/vision/stable/models.html#classification', 'segmentation': 'https://docs.pytorch.org/vision/stable/models.html#semantic-segmentation', 'detection': 'https://docs.pytorch.org/vision/stable/models.html#object-detection', 'video': 'https://docs.pytorch.org/vision/stable/models.html#video-classification', 'optical_flow': 'https://docs.pytorch.org/vision/stable/models.html#optical-flow'}
    compare = {'classification': '1000 类 logits，逐元素比对并要求 argmax 一致', 'segmentation': '`out` 分支的 logits 图（1x21xHxW），逐元素比对', 'detection': '网络部分的输出：骨干 + FPN + 检测头在全部 anchor 上的输出拼成一行（两阶段模型到 RPN 头为止），逐元素比对。后处理（分数阈值、NMS）的输出形状数据相关，torch.export 无法静态化，未包含', 'video': '400 类 logits，逐元素比对并要求 argmax 一致', 'optical_flow': '最后一轮迭代得到的光流场（1x2xHxW），逐元素比对'}
    notes = [('alexnet', '输入取 64x64：最后一个 max-pool 需要至少 2x2 的特征图。'), ('inception_v3', '输入取 80x80（文档最小 75）。'), ('vit_', '`image_size` 设为输入尺寸；torchvision 把 ViT 的分类头零初始化（logits 恒为 0），导出时给全零 Linear 一个小随机初始化。'), ('maxvit_t', '直接以 `MaxVit(...)` 构造 maxvit_t 的块配置，`partition_size=2`（默认 7 只能整除 224 输入的各级网格）。'), ('swin_', 'Swin 的相对位置偏置 `table[index]` 用注册的索引 buffer，torch-mlir 的 fx importer 拒绝；eval 下它是常量，导出时逐块预计算成普通 buffer。'), ('swin3d', '`torch.roll` 在时间维上位移为 0（时间窗覆盖整段 clip），会 lower 成 0 长度 slice + concat 并在 bufferize 后越界；导出时去掉零位移维度。'), ('s3d', '末尾 `AvgPool3d((2,7,7))` 是按 16x224x224 clip 设计的，改为全局平均池化以适应 16x64x64 输入。'), ('raft', '两帧堆叠为一个 2x3xHxW 输入；输入 128x128（特征图 /8 后相关金字塔需要 >=16）；`_iters` 次迭代更新。'), ('ssd300', '输入 300x300，检测头按此尺寸设计。'), ('regnet_y_128gf', '权重 2.5 GB，超过 PC 相对寻址范围：`split_weights.py` 把权重 blob 对半分到代码前后两个段（`tv.ld`），模拟器内存按权重大小自动给到 4 GB。'), ('vit_h_14', '权重 2.4 GB，超过 PC 相对寻址范围：`split_weights.py` 把权重 blob 对半分到代码前后两个段（`tv.ld`），模拟器内存按权重大小自动给到 4 GB。')]
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}_tv.s')): continue
        hw, kw = models.get(case, (32, '{}')); fam = fam_of(case)
        wpath = os.path.join(D, case, f'{case}_tv_weights.bin'); wsz = os.path.getsize(wpath) if os.path.exists(wpath) else None
        cb = os.path.join(D, case, f'{case}_tv_check.bin'); nin = nout = None
        if os.path.exists(cb):
            with open(cb, 'rb') as f:
                f.read(4); nin = struct.unpack('i', f.read(4))[0]; f.seek(8 + nin * 4); nout = struct.unpack('i', f.read(4))[0]
        cfg = {}
        try:
            import json; cfg = json.loads(kw)
        except Exception: pass
        T = cfg.get('_T'); it = cfg.get('_iters')
        if fam == 'video': inshape = f'1x3x{T}x{hw}x{hw}'
        elif fam == 'optical_flow': inshape = f'2x3x{hw}x{hw}'
        else: inshape = f'1x3x{hw}x{hw}'
        t = f'# {case}\n\n'
        t += f'torchvision `{case}`（{fam_cn[fam]}）在 Hwacha 上的一次前向，与 PyTorch 比对。文档：{fam_doc[fam]}\n\n'
        t += f'比对内容：{compare[fam]}。\n\n'
        t += '权重随机（固定种子，BatchNorm 给随机的 running 统计量使前向非退化），输入随机；同一组权重同时用于 PyTorch 参考与 Hwacha 构建。\n\n'
        ns = [n for k, n in notes if (case.startswith('vit_') if k == 'vit_' else k in case)]
        if ns: t += '## 本 case 的特殊处理\n\n' + ''.join(f'- {n}\n' for n in dict.fromkeys(ns)) + '\n'
        t += '## 形状与规模\n\n| | 值 |\n|---|---|\n' + f'| 输入 | {inshape}（{nin:,} 个 float）|\n' if nin else f'| 输入 | {inshape} |\n'
        if nout: t += f'| 输出元素数 | {nout:,} |\n'
        np_ = params.get(case) or nparams(case)
        if np_: t += f'| 参数量 | {np_:.1f}M |\n'
        if wsz: t += f'| 权重 blob | {wsz / 1048576:.0f} MB |\n'
        ck = {k: v for k, v in cfg.items() if not k.startswith('_')}
        if ck: t += f'| 构造参数 | `{ck}` |\n'
        if T: t += f'| 帧数 | {T} |\n'
        if it: t += f'| 光流迭代次数 | {it} |\n'
        t += '\n## 文件\n\n' + files_table(case, f'{case}_tv.s', [(f'`{case}_tv_weights.bin.S`', '权重的 `.incbin` 桩，按符号切分 blob；前半段放 `.weights_lo`、后半段放 `.weights_hi`（`split_weights.py`）'), (f'`{case}_tv_weights.bin`', f'权重 blob（不入 git，`make gen-{case}` 按固定种子逐字节重建）'), (f'`{case}_tv_check.bin`', '输入与 PyTorch 参考输出，host 用 `.incbin` 内嵌'), ('`tv_main.c`', '通用 host：调用 `net`，比对 max\\|diff\\|（容差 1e-4 + 1e-2·max\\|ref\\|）与 argmax（分类输出）'), ('`hwlib.s`', '卷积 / 池化库内核')] + ([('`HWMLIRFLAGS`', '本 case 需要的 hwacha-mlir 映射选项')] if flags(case) else []))
        t += '\n## 编译与运行\n\n```\nmake ' + case + '          # 编译 -> ' + case + '/' + case + '_tv.riscv\nmake ' + case + '.spike    # 在 Spike 上运行（内存按权重大小自动确定）\nmake gen-' + case + '      # 从 PyTorch 重新生成汇编、权重与参考\n```\n\n' + f'hwacha-mlir 映射：`{flags(case) or "默认（最内维为 lane）"}`。\n\n'
        t += '## Spike 结果\n\n' + fmt_res(R.get(case)) + '\n'
        write(case, t)

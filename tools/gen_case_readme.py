#!/usr/bin/env python3
"""Write a README.md into every case directory of torch-module / torch-function / torch-vision, from the
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

if d == 'torch-module':
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

elif d == 'torch-function':
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
            'myocyte': ('myocyte', 'kernel_gpu_opencl（group 0 / lane 0 跑 ECC 模型，group 1 / lane 0 跑三次 CaM 模型：一次 ODE 右端项求值）', '91 个方程，18 个参数', 'EQUATIONS=91 PARAMETERS=18')}
    issues = {'srad': 'srad_kernel 的最终 store 用一个已被复用的寄存器做索引（hwacha-cc 寄存器分配问题），Spike 上 STORE ACCESS FAULT；见 `../known-issues/README.md`。',
              'backprop': '16x16 的 work-group（256 个 work-item）超过 Hwacha 给这个内核的 maxvl（184），组被拆成两个 stripmine，组内 barrier 分隔的树形归约失效；见 `../known-issues/README.md`。',
              'myocyte': 'kernel 只是分发器，两个约千行的被调函数不内联时 hwacha-cc 不生成其代码，强制内联后又超出 64 个 vs 寄存器；见 `../known-issues/README.md`。'}
    for case in sorted(os.listdir(D)):
        if not os.path.isfile(os.path.join(D, case, f'{case}.s')): continue
        app, kern, size, defs = info.get(case, (case, '', '', ''))
        line = R.get(case, '')
        cyc = re.findall(r'([a-z_]+) (scalar|hwacha-cc): (\d+) cycles', line); verdicts = re.findall(r'([a-z_]+) (PASS|FAIL)', line)
        entries = sorted(set(re.findall(r'^\s*([A-Za-z_0-9]+_ct):', open(os.path.join(D, case, f'{case}.s')).read(), re.M)))
        t = f'# {case}\n\n'
        t += f'Rodinia `{app}` 的 OpenCL 内核在 Hwacha 上运行：**{kern}**。\n\n'
        mod = {'lavamd': '`NUMBER_THREADS` 的 #define 加了 #ifndef 以便由 Makefile 的 -D 覆盖', 'btree': 'kernel_gpu_opencl.cl 与 kernel_gpu_opencl_2.cl 合并为一个文件，第二个文件重复的结构体定义去掉、DEFAULT_ORDER_2 统一为 DEFAULT_ORDER', 'hybridsort': 'histogram1024.cl 与 bucketsort_kernels.cl 合并为一个文件', 'myocyte': 'kernel_ecc / kernel_cam 加了 __attribute__((always_inline))'}.get(case)
        t += f'内核文件 `{case}.cl` 是 Rodinia 3.1 的原版' + (f'，唯一改动：{mod}' if mod else '，未做修改') + f'；hwacha-cc 把每个 work-item 映射到一个 Hwacha lane，生成的入口是控制线程函数 `{"`, `".join(entries)}`，host 以 OpenCL host 传给 kernel 的同样参数调用它们。\n\n'
        t += f'问题规模：{size}（`{case}_main.c` 中 `{defs}`）。输入由固定种子的伪随机数生成；host 先在 Rocket 标量核上跑一个参考实现，再跑 Hwacha 内核，逐元素比对并打印两者的周期数。\n\n'
        extra = [(f'`{case}.cl`', 'Rodinia 原版 OpenCL 内核' + ('（lavaMD 的 NUMBER_THREADS 改为可由 -D 覆盖）' if case == 'lavamd' else '（两个 .cl 合并为一个文件）' if case in ('btree', 'hybridsort') else '（两个辅助函数加了 always_inline）' if case == 'myocyte' else '')), (f'`{case}_main.c`', '裸机 host：构造输入、标量参考、调用 Hwacha 内核、比对并打印 PASS/FAIL 与周期数'), ('`common.h`', '伪随机数、rdcycle、REPORT、NDRANGE1/2 启动宏、check_f、__errno 与陷阱桩')]
        if os.path.exists(os.path.join(D, case, f'{case}_ref.c')): extra.append((f'`{case}_ref.c`', '标量参考：同一 .cl 以 C 编译（OpenCL 限定符定义为空）'))
        if os.path.exists(os.path.join(D, case, 'main.h')): extra.append(('`main.h`', '内核 #include 的 Rodinia host 头文件的替身（fp、NUMBER_THREADS）'))
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

elif d == 'torch-vision':
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

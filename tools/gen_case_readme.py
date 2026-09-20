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

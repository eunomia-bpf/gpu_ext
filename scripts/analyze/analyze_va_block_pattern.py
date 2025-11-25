#!/usr/bin/env python3
"""
分析同一个 VA block 内的连续访问 pattern
在切换到下一个 VA block 之前，page_index 是如何变化的？
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data(filename):
    print(f"Loading {filename}...")
    df = pd.read_csv(filename)

    # Parse hex addresses
    def parse_hex(x):
        try:
            if pd.isna(x) or x == '' or x == '0x0':
                return 0
            return int(x, 16) if isinstance(x, str) else int(x)
        except:
            return 0

    df['va_start_int'] = df['va_start'].apply(parse_hex)
    print(f"Loaded {len(df):,} events, {df['va_start_int'].nunique():,} unique VA blocks")
    return df


def analyze_sequences(df):
    """找出所有连续访问同一 VA block 的序列"""
    print("\n分析连续访问序列...")

    sequences = []
    current_va = None
    current_seq = []

    for idx, row in df.iterrows():
        va = row['va_start_int']
        page_idx = row['page_index']

        if va == current_va:
            # 同一个 VA block，继续累积
            current_seq.append(page_idx)
        else:
            # 切换到新的 VA block
            if len(current_seq) > 0:
                sequences.append({
                    'va': current_va,
                    'pages': current_seq.copy(),
                    'length': len(current_seq)
                })
            current_va = va
            current_seq = [page_idx]

    # 别忘了最后一个序列
    if len(current_seq) > 0:
        sequences.append({
            'va': current_va,
            'pages': current_seq.copy(),
            'length': len(current_seq)
        })

    print(f"找到 {len(sequences):,} 个连续访问序列")
    return sequences


def analyze_patterns(sequences):
    """分析序列的 pattern"""
    print("\n" + "="*80)
    print("序列长度分析")
    print("="*80)

    lengths = [s['length'] for s in sequences]
    print(f"序列数量: {len(lengths):,}")
    print(f"最短序列: {min(lengths)}")
    print(f"最长序列: {max(lengths)}")
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"中位数长度: {np.median(lengths):.0f}")

    # 长度分布
    print("\n序列长度分布:")
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 1000]
    for i in range(len(bins)-1):
        count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
        pct = count / len(lengths) * 100
        bar = '#' * int(pct / 2)
        print(f"  {bins[i]:>3}-{bins[i+1]-1:<3}: {count:>8,} ({pct:>5.1f}%) {bar}")
    count = sum(1 for l in lengths if l >= bins[-1])
    pct = count / len(lengths) * 100
    print(f"  {bins[-1]:>3}+   : {count:>8,} ({pct:>5.1f}%)")

    print("\n" + "="*80)
    print("序列内访问 Pattern 分析")
    print("="*80)

    # 分析长度 >= 2 的序列
    multi_seqs = [s for s in sequences if s['length'] >= 2]
    print(f"\n长度>=2 的序列数: {len(multi_seqs):,}")

    # 分析 page_index 的变化
    all_diffs = []
    pattern_counts = defaultdict(int)

    for seq in multi_seqs:
        pages = seq['pages']
        diffs = [pages[i+1] - pages[i] for i in range(len(pages)-1)]
        all_diffs.extend(diffs)

        # 判断 pattern 类型
        if all(d == 0 for d in diffs):
            pattern_counts['same_page'] += 1
        elif all(d > 0 for d in diffs):
            pattern_counts['increasing'] += 1
        elif all(d < 0 for d in diffs):
            pattern_counts['decreasing'] += 1
        elif all(d == diffs[0] for d in diffs) and diffs[0] != 0:
            pattern_counts['constant_stride'] += 1
        else:
            pattern_counts['mixed'] += 1

    print("\nPattern 类型分布:")
    total = len(multi_seqs)
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        desc = {
            'same_page': '相同页面 (重复访问)',
            'increasing': '递增 (顺序向前)',
            'decreasing': '递减 (顺序向后)',
            'constant_stride': '固定步长',
            'mixed': '混合/随机'
        }.get(pattern, pattern)
        print(f"  {desc:<25}: {count:>8,} ({pct:>5.1f}%) {bar}")

    print("\n" + "="*80)
    print("Page Index 变化 (diff) 分析")
    print("="*80)

    all_diffs = np.array(all_diffs)
    print(f"\n总 diff 数: {len(all_diffs):,}")
    print(f"diff 范围: {all_diffs.min()} 到 {all_diffs.max()}")
    print(f"平均 diff: {all_diffs.mean():.2f}")
    print(f"中位数 diff: {np.median(all_diffs):.0f}")

    # diff 分布
    print("\ndiff 分布:")
    zero_count = np.sum(all_diffs == 0)
    pos_count = np.sum(all_diffs > 0)
    neg_count = np.sum(all_diffs < 0)
    print(f"  diff = 0 (相同页): {zero_count:>10,} ({zero_count/len(all_diffs)*100:>5.1f}%)")
    print(f"  diff > 0 (向前):   {pos_count:>10,} ({pos_count/len(all_diffs)*100:>5.1f}%)")
    print(f"  diff < 0 (向后):   {neg_count:>10,} ({neg_count/len(all_diffs)*100:>5.1f}%)")

    # 常见 diff 值
    print("\n最常见的 diff 值 (Top 20):")
    unique, counts = np.unique(all_diffs, return_counts=True)
    sorted_idx = np.argsort(-counts)[:20]
    for i, idx in enumerate(sorted_idx):
        d = unique[idx]
        c = counts[idx]
        pct = c / len(all_diffs) * 100
        print(f"  {i+1:>2}. diff={d:>4}: {c:>10,} ({pct:>5.2f}%)")

    # 步长范围分布
    print("\n|diff| 步长范围分布:")
    abs_diffs = np.abs(all_diffs)
    ranges = [(0, 0, '=0 (相同)'), (1, 1, '=1 (相邻)'), (2, 16, '2-16 (近距离)'),
              (17, 64, '17-64 (中距离)'), (65, 256, '65-256 (远距离)'), (257, 512, '257+ (跨半块)')]
    for lo, hi, desc in ranges:
        count = np.sum((abs_diffs >= lo) & (abs_diffs <= hi))
        pct = count / len(abs_diffs) * 100
        bar = '#' * int(pct / 2)
        print(f"  {desc:<20}: {count:>10,} ({pct:>5.1f}%) {bar}")

    return all_diffs, sequences


def show_example_sequences(sequences, n=10):
    """展示一些典型的序列例子"""
    print("\n" + "="*80)
    print(f"典型序列示例 (随机选取长度>=3的序列)")
    print("="*80)

    long_seqs = [s for s in sequences if s['length'] >= 3]
    if len(long_seqs) == 0:
        print("没有长度>=3的序列")
        return

    # 随机选择
    np.random.seed(42)
    sample_indices = np.random.choice(len(long_seqs), min(n, len(long_seqs)), replace=False)

    for i, idx in enumerate(sample_indices):
        seq = long_seqs[idx]
        pages = seq['pages']
        diffs = [pages[j+1] - pages[j] for j in range(len(pages)-1)]

        print(f"\n序列 #{i+1} (长度={len(pages)}):")
        print(f"  VA: 0x{seq['va']:x}")
        if len(pages) <= 20:
            print(f"  Pages: {pages}")
            print(f"  Diffs: {diffs}")
        else:
            print(f"  Pages (前20): {pages[:20]}...")
            print(f"  Diffs (前19): {diffs[:19]}...")

        # 判断 pattern
        if all(d == 0 for d in diffs):
            pattern = "重复访问同一页"
        elif all(d > 0 for d in diffs):
            pattern = "递增顺序"
        elif all(d < 0 for d in diffs):
            pattern = "递减顺序"
        elif len(set(diffs)) == 1:
            pattern = f"固定步长 (stride={diffs[0]})"
        else:
            pattern = "混合模式"
        print(f"  Pattern: {pattern}")


def plot_patterns(all_diffs, sequences, output_prefix='/tmp/va_block_pattern'):
    """绘制 pattern 分析图"""
    print(f"\n生成图表...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: diff 分布直方图
    ax1 = axes[0, 0]
    # 限制范围避免极端值
    diffs_clipped = np.clip(all_diffs, -100, 100)
    ax1.hist(diffs_clipped, bins=201, range=(-100, 100), edgecolor='none', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Page Index Diff')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Page Index Diff Distribution (clipped to [-100, 100])')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='diff=0')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: 序列长度分布
    ax2 = axes[0, 1]
    lengths = [s['length'] for s in sequences]
    ax2.hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Consecutive Access Sequence Length Distribution')
    ax2.axvline(np.median(lengths), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(lengths):.0f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: |diff| 分布 (绝对步长)
    ax3 = axes[1, 0]
    abs_diffs = np.abs(all_diffs)
    abs_diffs_clipped = np.clip(abs_diffs, 0, 200)
    ax3.hist(abs_diffs_clipped, bins=100, range=(0, 200), edgecolor='none', alpha=0.7, color='green')
    ax3.set_xlabel('|Page Index Diff| (Stride)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Absolute Stride Distribution (clipped to [0, 200])')
    ax3.axvline(np.median(abs_diffs), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(abs_diffs):.0f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: 随机选择几个长序列可视化
    ax4 = axes[1, 1]
    long_seqs = [s for s in sequences if s['length'] >= 10]
    if len(long_seqs) > 0:
        np.random.seed(42)
        sample_seqs = [long_seqs[i] for i in np.random.choice(len(long_seqs), min(5, len(long_seqs)), replace=False)]

        for i, seq in enumerate(sample_seqs):
            pages = seq['pages'][:50]  # 最多显示前50个
            ax4.plot(range(len(pages)), pages, marker='.', markersize=3, alpha=0.7,
                    linewidth=1, label=f'Seq {i+1} (len={seq["length"]})')

        ax4.set_xlabel('Access Order (within sequence)')
        ax4.set_ylabel('Page Index')
        ax4.set_title('Example Sequences: Page Index vs Access Order')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 512)

    plt.tight_layout()
    output_file = f'{output_prefix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else '/tmp/prefetch_combined.csv'

    df = load_data(filename)
    sequences = analyze_sequences(df)
    all_diffs, sequences = analyze_patterns(sequences)
    show_example_sequences(sequences, n=10)
    plot_patterns(all_diffs, sequences)

    print("\n" + "="*80)
    print("总结")
    print("="*80)

    # 总结
    multi_seqs = [s for s in sequences if s['length'] >= 2]
    all_diffs = np.array(all_diffs)

    zero_pct = np.sum(all_diffs == 0) / len(all_diffs) * 100
    small_stride_pct = np.sum(np.abs(all_diffs) <= 16) / len(all_diffs) * 100

    print(f"\n1. 连续访问序列:")
    print(f"   - 平均序列长度: {np.mean([s['length'] for s in sequences]):.1f}")
    print(f"   - 这意味着平均每 {np.mean([s['length'] for s in sequences]):.1f} 次 prefetch 后切换到新的 VA block")

    print(f"\n2. 同一 VA block 内的访问模式:")
    print(f"   - {zero_pct:.1f}% 的访问是同一页面 (diff=0)")
    print(f"   - {small_stride_pct:.1f}% 的访问是近距离 (|diff| <= 16)")

    if zero_pct > 50:
        print(f"\n   → 结论: 主要是重复访问同一页面，可能是热点访问模式")
    elif small_stride_pct > 70:
        print(f"\n   → 结论: 主要是顺序或近距离访问，适合激进 prefetch")
    else:
        print(f"\n   → 结论: 访问模式较随机，可能需要自适应 prefetch")


if __name__ == '__main__':
    main()

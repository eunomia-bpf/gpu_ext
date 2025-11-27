# 使用 cflow 和 doxygen 生成 UVM 调用图

## 安装工具

```bash
# 运行安装脚本
bash /tmp/install_callgraph_tools.sh

# 或手动安装
sudo apt-get update
sudo apt-get install -y cflow doxygen graphviz
```

---

## 方法1: cflow (快速文本调用图)

### 基本用法

```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

# 查看特定函数的调用关系
cflow --tree --main=uvm_parent_gpu_service_replayable_faults \
      uvm_gpu_replayable_faults.c
```

### 常用选项

```bash
# 1. 查看特定函数，限制深度
cflow --tree --main=uvm_parent_gpu_service_replayable_faults \
      --depth=3 \
      uvm_gpu_replayable_faults.c

# 2. 显示行号
cflow --tree --number \
      --main=uvm_parent_gpu_service_replayable_faults \
      uvm_gpu_replayable_faults.c

# 3. 反向调用图（谁调用了这个函数）
cflow --reverse --tree \
      --main=uvm_parent_gpu_service_replayable_faults \
      uvm_gpu_replayable_faults.c

# 4. 省略参数，更简洁
cflow --tree --omit-arguments \
      --main=uvm_parent_gpu_service_replayable_faults \
      uvm_gpu_replayable_faults.c

# 5. 生成完整文件的调用图
cflow --tree \
      uvm_gpu_replayable_faults.c > callgraph_full.txt

# 6. 多文件分析
cflow --tree --main=uvm_parent_gpu_service_replayable_faults \
      uvm_gpu_replayable_faults.c uvm_va_block.c uvm_perf_thrashing.c
```

### 实用示例

#### 示例1: 分析 Fault Service 调用链
```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

cflow --tree --number --depth=5 \
      --main=uvm_parent_gpu_service_replayable_faults \
      uvm_gpu_replayable_faults.c \
      > cflow_fault_service.txt

# 查看结果
less cflow_fault_service.txt
```

#### 示例2: 分析 Thrashing 检测
```bash
cflow --tree --number --depth=4 \
      --main=uvm_perf_thrashing_get_hint \
      uvm_perf_thrashing.c \
      > cflow_thrashing.txt
```

#### 示例3: 找到谁调用了 replay 函数
```bash
cflow --reverse --tree \
      uvm_gpu_replayable_faults.c | \
      grep -A 10 "push_replay_on_parent_gpu" \
      > cflow_replay_callers.txt
```

#### 示例4: 生成所有函数的调用关系
```bash
# 生成所有函数列表
cflow --omit-arguments *.c > cflow_all_functions.txt

# 只看 uvm 开头的函数
cflow *.c | grep "^uvm_" > cflow_uvm_functions.txt
```

### cflow 输出格式说明

```
uvm_parent_gpu_service_replayable_faults() <void uvm_parent_gpu_service_replayable_faults (uvm_parent_gpu_t *parent_gpu) at uvm_gpu_replayable_faults.c:2906> (R):
    fetch_fault_buffer_entries() <int fetch_fault_buffer_entries (...) at uvm_gpu_replayable_faults.c:844>:
        fault_buffer_read_get()
        fault_buffer_read_put()
    service_fault_batch() <int service_fault_batch (...) at uvm_gpu_replayable_faults.c:2232>:
        service_managed_fault_in_block()
```

说明：
- `<>` 内是函数签名和位置
- `(R)` 表示递归调用
- 缩进表示调用层级

---

## 方法2: doxygen (完整HTML文档 + 可视化调用图)

### 创建配置文件

```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

# 生成默认配置
doxygen -g Doxyfile
```

### 编辑 Doxyfile (推荐配置)

创建优化的配置文件：

```bash
cat > Doxyfile.uvm << 'EOF'
# 项目信息
PROJECT_NAME           = "NVIDIA UVM Driver"
PROJECT_BRIEF          = "Unified Virtual Memory Driver - Policy Analysis"
PROJECT_NUMBER         = "575.57.08"

# 输出目录
OUTPUT_DIRECTORY       = doxygen_output

# 输入源码
INPUT                  = .
FILE_PATTERNS          = *.c *.h
RECURSIVE              = NO
EXCLUDE_PATTERNS       = *test*.c

# 提取选项
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = YES
EXTRACT_STATIC         = YES
EXTRACT_LOCAL_CLASSES  = YES

# 调用图选项 (关键！)
HAVE_DOT               = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
DOT_IMAGE_FORMAT       = svg
INTERACTIVE_SVG        = YES
DOT_GRAPH_MAX_NODES    = 100
MAX_DOT_GRAPH_DEPTH    = 5

# 其他图形
INCLUDE_GRAPH          = YES
INCLUDED_BY_GRAPH      = YES
COLLABORATION_GRAPH    = YES
CLASS_GRAPH            = YES

# 输出格式
GENERATE_HTML          = YES
GENERATE_LATEX         = NO
GENERATE_RTF           = NO
GENERATE_MAN           = NO

# HTML选项
HTML_OUTPUT            = html
HTML_FILE_EXTENSION    = .html
HTML_DYNAMIC_SECTIONS  = YES
GENERATE_TREEVIEW      = YES

# 搜索引擎
SEARCHENGINE           = YES

# 源码浏览
SOURCE_BROWSER         = YES
INLINE_SOURCES         = NO
STRIP_CODE_COMMENTS    = NO
REFERENCED_BY_RELATION = YES
REFERENCES_RELATION    = YES

# 优化
OPTIMIZE_OUTPUT_FOR_C  = YES
SHOW_INCLUDE_FILES     = YES
SORT_MEMBER_DOCS       = NO

# Graphviz调优
DOT_CLEANUP            = YES
DOT_MULTI_TARGETS      = YES
TEMPLATE_RELATIONS     = YES
EOF
```

### 生成文档

```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

# 运行 doxygen
doxygen Doxyfile.uvm

# 查看进度
tail -f doxygen_output/doxygen_warnings.txt
```

### 查看生成的调用图

```bash
# 方法1: 用浏览器打开
firefox doxygen_output/html/index.html

# 方法2: 用Chrome
google-chrome doxygen_output/html/index.html

# 方法3: 用Python简单HTTP服务器
cd doxygen_output/html
python3 -m http.server 8080
# 然后访问 http://localhost:8080
```

### 在 doxygen HTML 中查找调用图

1. 打开 `doxygen_output/html/index.html`
2. 点击顶部的 **"Files"** 或 **"Globals"**
3. 搜索函数名，例如 `uvm_parent_gpu_service_replayable_faults`
4. 点击函数名进入详细页面
5. 向下滚动查看：
   - **Call Graph** (这个函数调用了谁)
   - **Caller Graph** (谁调用了这个函数)
6. SVG图表是交互式的，可以点击节点跳转

### 高级配置：只分析特定文件

如果只想分析 fault 相关的文件：

```bash
cat > Doxyfile.fault << 'EOF'
PROJECT_NAME           = "UVM Fault Handling"
OUTPUT_DIRECTORY       = doxygen_fault
INPUT                  = uvm_gpu_replayable_faults.c \
                         uvm_gpu_replayable_faults.h \
                         uvm_va_block.c \
                         uvm_va_block.h
RECURSIVE              = NO
EXTRACT_ALL            = YES
HAVE_DOT               = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
DOT_IMAGE_FORMAT       = svg
INTERACTIVE_SVG        = YES
MAX_DOT_GRAPH_DEPTH    = 4
GENERATE_LATEX         = NO
EOF

doxygen Doxyfile.fault
firefox doxygen_fault/html/index.html
```

---

## 实战示例

### 任务1: 分析 Fault Service 完整流程

```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

# Step 1: 用 cflow 快速查看文本调用图
cflow --tree --number --depth=5 \
      --main=uvm_parent_gpu_service_replayable_faults \
      uvm_gpu_replayable_faults.c \
      | tee cflow_fault_service.txt \
      | less

# Step 2: 用 doxygen 生成可视化
cat > Doxyfile.fault_only << 'EOF'
PROJECT_NAME     = "Fault Service Analysis"
OUTPUT_DIRECTORY = doxygen_fault_service
INPUT            = uvm_gpu_replayable_faults.c uvm_gpu_replayable_faults.h
EXTRACT_ALL      = YES
HAVE_DOT         = YES
CALL_GRAPH       = YES
CALLER_GRAPH     = YES
MAX_DOT_GRAPH_DEPTH = 5
GENERATE_LATEX   = NO
EOF

doxygen Doxyfile.fault_only
firefox doxygen_fault_service/html/index.html
```

### 任务2: 找到所有 Policy 决策点

```bash
# Step 1: 用 cflow 列出所有函数
cflow --omit-arguments *.c | grep -i "policy\|replay\|thrash" > policy_functions.txt

# Step 2: 对每个 policy 函数生成调用图
while read func; do
    func_name=$(echo $func | awk '{print $1}' | sed 's/()//')
    echo "=== Analyzing $func_name ==="
    cflow --tree --depth=3 --main=$func_name *.c > "cflow_${func_name}.txt" 2>/dev/null
done < policy_functions.txt
```

### 任务3: 对比不同 Replay Policy 的实现

```bash
# 查找所有 replay 相关函数
cflow --reverse *.c | grep -i replay | head -30

# 生成 replay 函数族的调用图
cflow --tree \
      --main=push_replay_on_parent_gpu \
      --depth=4 \
      uvm_gpu_replayable_faults.c \
      > cflow_replay_implementation.txt
```

---

## cflow vs doxygen 对比

| 特性 | cflow | doxygen |
|------|-------|---------|
| **速度** | 非常快 (秒级) | 较慢 (分钟级) |
| **输出** | 纯文本 | HTML + SVG图 |
| **可视化** | 无 | 交互式SVG |
| **搜索** | grep | Web搜索 |
| **调用深度** | 无限制 | 可配置 (建议≤5) |
| **反向查找** | 支持 (`--reverse`) | 支持 (Caller Graph) |
| **跨文件** | 支持 | 支持 |
| **学习曲线** | 低 | 中 |
| **适用场景** | 快速查看、脚本处理 | 详细分析、团队共享 |

## 推荐工作流

### 快速查看（推荐 cflow）
```bash
# 1. 快速查看某函数的调用
cflow --tree --depth=3 --main=FUNCTION_NAME file.c | less

# 2. 找调用者
cflow --reverse --tree file.c | grep FUNCTION_NAME

# 3. 生成简单报告
cflow --tree file.c > report.txt
```

### 深入分析（推荐 doxygen）
```bash
# 1. 生成完整文档
doxygen Doxyfile.uvm

# 2. 在浏览器中交互式探索
firefox doxygen_output/html/index.html

# 3. 查看可视化调用图
# 在网页中点击函数 → 查看 Call Graph/Caller Graph
```

### 组合使用
```bash
# 1. 先用 cflow 快速定位
cflow --tree --depth=2 *.c | grep -i "policy" > targets.txt

# 2. 找到目标函数后，用 doxygen 深入分析
# 编辑 Doxyfile，只包含相关文件
# 然后生成详细文档
```

---

## 故障排查

### cflow 找不到函数
```bash
# 问题：cflow 只分析静态函数可见性
# 解决：包含头文件或多个源文件

cflow --tree --main=FUNCTION_NAME \
      file.c related_file.c header.h
```

### doxygen 调用图不显示
```bash
# 检查配置
grep "HAVE_DOT\|CALL_GRAPH" Doxyfile

# 应该看到:
# HAVE_DOT = YES
# CALL_GRAPH = YES
# CALLER_GRAPH = YES

# 检查 graphviz 是否安装
which dot
dot -V
```

### 生成的图太大/太小
```bash
# 在 Doxyfile 中调整:
DOT_GRAPH_MAX_NODES    = 50   # 减少节点数
MAX_DOT_GRAPH_DEPTH    = 3    # 减少深度

# 或针对 cflow:
cflow --tree --depth=3 ...    # 限制深度
```

---

## 输出示例

### cflow 输出示例
```
uvm_parent_gpu_service_replayable_faults() <void uvm_parent_gpu_service_replayable_faults (uvm_parent_gpu_t *) at uvm_gpu_replayable_faults.c:2906>:
    uvm_tracker_init() <void uvm_tracker_init (uvm_tracker_t *) at uvm_tracker.c:45>
    fetch_fault_buffer_entries() <int fetch_fault_buffer_entries (...) at uvm_gpu_replayable_faults.c:844>:
        fault_buffer_read_get() <NvU32 at uvm_gpu_replayable_faults.c:652>
        fault_buffer_read_put() <NvU32 at uvm_gpu_replayable_faults.c:658>
    service_fault_batch() <NV_STATUS service_fault_batch (...) at uvm_gpu_replayable_faults.c:2232>:
        service_managed_fault_in_block()
        uvm_va_block_make_resident()
    push_replay_on_parent_gpu() <NV_STATUS push_replay_on_parent_gpu (...) at uvm_gpu_replayable_faults.c:503>
```

### doxygen HTML 示例

打开后会看到类似这样的界面：

```
[File List] [Functions] [Search]

uvm_parent_gpu_service_replayable_faults
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Definition: uvm_gpu_replayable_faults.c:2906

[Call Graph]  [Caller Graph]  [Source Code]

Calls:
  • fetch_fault_buffer_entries()
  • service_fault_batch()
  • push_replay_on_parent_gpu()

Called by:
  • uvm_isr_replayable_faults()
  • uvm_fault_service_work()
```

点击 [Call Graph] 会显示交互式的 SVG 调用图。

---

## 总结

**快速查看用 cflow**：
```bash
cflow --tree --main=FUNCTION file.c | less
```

**详细分析用 doxygen**：
```bash
doxygen Doxyfile.uvm
firefox doxygen_output/html/index.html
```

两者结合使用效果最好！

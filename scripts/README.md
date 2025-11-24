# GPU Chunk Trace Analysis Scripts

This directory contains scripts for analyzing GPU memory chunk lifecycle traces captured by the `chunk_trace` tool.

## Scripts

### 1. analyze_chunk_trace.py

**Purpose**: General analysis of chunk lifecycle events

**Usage**:
```bash
python3 analyze_chunk_trace.py <csv_file>
```

**Analysis includes**:
- Basic statistics (total events, time range, unique chunks)
- Hook type distribution (ACTIVATE, POPULATE, DEPOPULATE, EVICTION_PREPARE)
- VA block tracking coverage
- Chunk lifecycle patterns (activations per chunk, populates per chunk)
- Chunk lifetime statistics

**Example output**:
```
CHUNK TRACE ANALYSIS
================================================================================
Total events:              421,533
Unique chunks tracked:     15,791
ACTIVATE                35,777 (  8.5%)
POPULATE              349,982 ( 83.0%)
...
```

### 2. analyze_va_blocks.py

**Purpose**: Deep analysis of VA block mapping patterns

**Usage**:
```bash
python3 analyze_va_blocks.py <csv_file>
```

**Analysis includes**:
- VA block statistics (total blocks, chunks mapped, unique ranges)
- VA block size distribution
- Chunks per VA block distribution
- Most active VA blocks (by event count)
- Detection of VA range aliasing

**Example output**:
```
VA BLOCK MAPPING ANALYSIS
================================================================================
Total VA blocks:           24,082
VA block size (all):       2.0 MB
Chunks per VA block:       Mean 1.84, Median 2.0
TOP 10 MOST ACTIVE VA BLOCKS...
```

## Data Format

The scripts analyze CSV files produced by `chunk_trace` tool with the following format:

```csv
time_ms,hook_type,cpu,chunk_addr,list_addr,va_block,va_start,va_end,va_page_index
0,POPULATE,1,0xffffcfd7df464c38,0xffff8a0a2ebd3a58,0xffff8a0b391ff350,0x7870e8a00000,0x7870e8bfffff,0
...
```

**Fields**:
- `time_ms`: Elapsed time in milliseconds since trace start
- `hook_type`: BPF hook type (ACTIVATE, POPULATE, DEPOPULATE, EVICTION_PREPARE)
- `cpu`: CPU core number
- `chunk_addr`: Physical address of GPU chunk
- `list_addr`: Address of list head (for ACTIVATE/POPULATE/DEPOPULATE)
- `va_block`: Virtual address block pointer (NULL if not available)
- `va_start`: VA block start address
- `va_end`: VA block end address
- `va_page_index`: Page index within VA block

## Running the Trace Tool

### Basic trace (5 seconds):
```bash
cd /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src
sudo timeout 5 ./chunk_trace > trace.csv 2>&1
```

### Analyze the trace:
```bash
cd /home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/scripts
python3 analyze_chunk_trace.py ../path/to/trace.csv
python3 analyze_va_blocks.py ../path/to/trace.csv
```

## Key Findings

From our analysis of 5-second traces:

1. **Hook Distribution**:
   - POPULATE dominates: ~83-88% of all events
   - ACTIVATE and EVICTION_PREPARE occur in ~1:1 ratio (~8-9% each)
   - DEPOPULATE hook never fires (blocked by `chunk_is_in_eviction()` check)

2. **VA Block Mapping**:
   - 100% VA block tracking coverage with BPF CO-RE
   - All VA blocks are 2MB in size (standard VA block size)
   - Average 1.84 chunks per VA block
   - 84% of VA blocks have exactly 2 chunks, 16% have 1 chunk

3. **Chunk Lifecycle**:
   - Most chunks experience 2 ACTIVATEs during the trace window
   - Chunks typically have 15-30 POPULATE events
   - Mean chunk lifetime: ~4 seconds (within 5-second trace window)

4. **VA Range Aliasing**:
   - Same VA range can have multiple VA block pointer instances
   - Indicates VA block structure reuse/recycling by UVM driver

## Implementation Details

The `chunk_trace` tool uses:
- **BPF kprobes** to hook into UVM driver functions
- **BPF CO-RE** (Compile Once - Run Everywhere) to access kernel structures
- **Ring buffer** for efficient event streaming to userspace
- **CSV output** for easy post-processing

## Future Analysis Ideas

See `FURTHER_ANALYSIS_IDEAS.md` in the parent directory for additional analysis directions.

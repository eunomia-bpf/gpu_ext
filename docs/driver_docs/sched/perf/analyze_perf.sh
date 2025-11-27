#!/bin/bash

# Analyze perf results and create a comparison table

OUTPUT_DIR="/home/yunwei37/workspace/gpu/open-gpu-kernel-modules/docs/sched"
ANALYSIS="$OUTPUT_DIR/perf_analysis.md"

{
    echo "# Perf Profile Analysis: Device vs UVM Mode"
    echo ""
    echo "Generated: $(date)"
    echo ""
    echo "## Test Configuration"
    echo "- Kernel: seq_stream"
    echo "- Size Factor: 0.1 (3210 MB working set)"
    echo "- Modes: device, uvm"
    echo "- Iterations: 100, 1000, 10000"
    echo ""

    echo "## Summary Table"
    echo ""
    echo "### Device Mode - Top Functions by Iteration Count"
    echo ""
    echo "| Iterations | osDevReadReg032 | uvm_pmm_devmem_* | intel_idle | Other GPU |"
    echo "|------------|-----------------|------------------|------------|-----------|"

    for iter in 100 1000 10000; do
        FILE="$OUTPUT_DIR/perf_device_${iter}.txt"
        if [ -f "$FILE" ]; then
            osdev=$(grep "osDevReadReg032" "$FILE" | head -1 | awk '{print $1}')
            uvm_pmm=$(grep "uvm_pmm_devmem" "$FILE" | head -1 | awk '{print $1}')
            idle=$(grep "intel_idle" "$FILE" | head -1 | awk '{print $1}')
            other=$(grep -E "kgsp|rpc" "$FILE" | head -1 | awk '{print $1}')
            echo "| $iter | $osdev | $uvm_pmm | $idle | $other |"
        fi
    done

    echo ""
    echo "### UVM Mode - Top Functions by Iteration Count"
    echo ""
    echo "| Iterations | osDevReadReg032 | uvm_va_block_* | uvm_pmm_devmem_* | fault_service |"
    echo "|------------|-----------------|----------------|------------------|---------------|"

    for iter in 100 1000 10000; do
        FILE="$OUTPUT_DIR/perf_uvm_${iter}.txt"
        if [ -f "$FILE" ]; then
            osdev=$(grep "osDevReadReg032" "$FILE" | head -1 | awk '{print $1}')
            va_block=$(grep "uvm_va_block_service" "$FILE" | head -1 | awk '{print $1}')
            uvm_pmm=$(grep "uvm_pmm_devmem" "$FILE" | head -1 | awk '{print $1}')
            fault=$(grep "service_fault" "$FILE" | head -1 | awk '{print $1}')
            echo "| $iter | $osdev | $va_block | $uvm_pmm | $fault |"
        fi
    done

    echo ""
    echo "## Detailed UVM-Related Functions"
    echo ""

    for mode in device uvm; do
        echo "### Mode: $mode"
        echo ""
        for iter in 100 1000 10000; do
            FILE="$OUTPUT_DIR/perf_${mode}_${iter}.txt"
            if [ -f "$FILE" ]; then
                echo "#### Iterations: $iter"
                echo ""
                echo "\`\`\`"
                grep -E "^[[:space:]]*[0-9]+\.[0-9]+%.*uvm_" "$FILE" | head -15
                echo "\`\`\`"
                echo ""
            fi
        done
    done

    echo "## Key Observations"
    echo ""
    echo "### Initialization Overhead"
    echo ""
    echo "Both modes show significant initialization overhead (osDevReadReg032) which includes:"
    echo "- Fault buffer setup (kgmmuClientShadowFaultBufferRegister)"
    echo "- GPU registration (uvm_api_register_gpu)"
    echo "- Channel creation and destruction"
    echo ""
    echo "### UVM-Specific Overhead"
    echo ""
    echo "UVM mode shows additional functions for page fault handling:"
    echo "- \`uvm_va_block_service_locked\` - VA block fault servicing"
    echo "- \`uvm_va_block_service_finish\` - Fault handling completion"
    echo "- \`service_fault_batch\` - Batch fault processing"
    echo "- \`replayable_faults_isr_bottom_half\` - Interrupt handling"
    echo ""
    echo "### Scaling Behavior"
    echo ""
    echo "As iterations increase:"
    echo "- Device mode: Initialization overhead percentage decreases"
    echo "- UVM mode: Page fault handling becomes more visible in profile"

} > "$ANALYSIS"

echo "Analysis saved to: $ANALYSIS"
cat "$ANALYSIS"

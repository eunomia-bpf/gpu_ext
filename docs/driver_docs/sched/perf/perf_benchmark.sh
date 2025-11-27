#!/bin/bash

# Perf benchmark script for uvmbench
# Generates perf profiles for different iteration counts and modes

OUTPUT_DIR="/home/yunwei37/workspace/gpu/open-gpu-kernel-modules/docs/sched"
PERF_CMD="/home/yunwei37/workspace/gpu/linux/tools/perf/perf"
UVMBENCH="/home/yunwei37/workspace/gpu/co-processor-demo/memory/micro/uvmbench"

# Iteration counts to test
ITERATIONS=(100 1000 10000)

# Modes to test
MODES=("device" "uvm")

echo "Starting perf benchmark..."
echo "Output directory: $OUTPUT_DIR"
echo ""

for mode in "${MODES[@]}"; do
    for iter in "${ITERATIONS[@]}"; do
        echo "=============================================="
        echo "Testing: mode=$mode, iterations=$iter"
        echo "=============================================="

        # Output files
        PERF_DATA="$OUTPUT_DIR/perf_${mode}_${iter}.data"
        PERF_TXT="$OUTPUT_DIR/perf_${mode}_${iter}.txt"

        # Run perf record
        echo "Recording perf data..."
        sudo $PERF_CMD record -g -a -o "$PERF_DATA" -- \
            $UVMBENCH --kernel=seq_stream --mode=$mode --size_factor=0.1 --iterations=$iter 2>&1

        # Generate report
        echo "Generating report..."
        sudo $PERF_CMD report -i "$PERF_DATA" --stdio --no-children -g folded > "$PERF_TXT" 2>&1

        # Extract summary (top 50 lines with overhead)
        echo ""
        echo "Top functions:"
        grep -E "^[[:space:]]*[0-9]+\.[0-9]+%" "$PERF_TXT" | head -20

        echo ""
        echo "UVM-related functions:"
        grep -E "(uvm|fault)" "$PERF_TXT" | head -10

        echo ""
        echo "Output saved to: $PERF_TXT"
        echo ""
    done
done

# Generate summary report
SUMMARY="$OUTPUT_DIR/summary.txt"
echo "=============================================="
echo "Generating summary report: $SUMMARY"
echo "=============================================="

{
    echo "Perf Benchmark Summary"
    echo "======================"
    echo "Generated: $(date)"
    echo ""

    for mode in "${MODES[@]}"; do
        echo ""
        echo "========================================"
        echo "Mode: $mode"
        echo "========================================"

        for iter in "${ITERATIONS[@]}"; do
            PERF_TXT="$OUTPUT_DIR/perf_${mode}_${iter}.txt"

            echo ""
            echo "--- Iterations: $iter ---"
            echo ""
            echo "Top 15 functions:"
            grep -E "^[[:space:]]*[0-9]+\.[0-9]+%" "$PERF_TXT" | head -15

            echo ""
            echo "UVM/Fault related:"
            grep -E "(uvm_|fault|nvUvm)" "$PERF_TXT" | grep -E "^[[:space:]]*[0-9]" | head -10
        done
    done
} > "$SUMMARY"

echo "Summary saved to: $SUMMARY"
echo ""
echo "All done!"

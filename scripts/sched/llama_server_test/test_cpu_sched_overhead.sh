#!/bin/bash
# CPU Scheduler Overhead Test for llama-server
# Tests llama-server performance with/without tracing and various noisy neighbors

set -e

# Configuration
LLAMA_SERVER="/home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp/build/bin/llama-server"
TRACE_TOOL="/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/tools/cuda_sched_trace"
DATASET_PATH="/home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
LLAMA_WORKDIR="/home/yunwei37/workspace/gpu/schedcp/workloads/llama.cpp"

# Get script directory for saving results
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DATE="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${SCRIPT_DIR}/results/${TEST_DATE}"

# Benchmark parameters - increased for stability
NUM_PROMPTS=200           # Increased from 100 for more stable results
NUM_RUNS=3                # Run each scenario 3 times for averaging
WARMUP_PROMPTS=10         # Warmup prompts before actual benchmark
REQUEST_RATE=1
MAX_CONCURRENCY=1
SERVER_PORT=8013
SERVER_WAIT=30            # seconds to wait for server to load model

mkdir -p "$OUTPUT_DIR"

echo "==================================================================="
echo "CPU Scheduler Overhead Test - llama-server"
echo "==================================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to start llama-server
start_llama_server() {
    local pinned=$1
    echo "  Starting llama-server..."

    if [ "$pinned" = "pinned" ]; then
        taskset -c 0-3 nice -n -10 "$LLAMA_SERVER" --gpt-oss-20b-default -c 65536 &
    else
        "$LLAMA_SERVER" --gpt-oss-20b-default -c 65536 &
    fi
    LLAMA_PID=$!

    echo "  Waiting ${SERVER_WAIT}s for model to load..."
    sleep "$SERVER_WAIT"

    # Verify server is running
    if ! kill -0 $LLAMA_PID 2>/dev/null; then
        echo "  ERROR: llama-server failed to start"
        return 1
    fi
    echo "  llama-server running (PID: $LLAMA_PID)"
}

# Function to stop llama-server
stop_llama_server() {
    if [ -n "$LLAMA_PID" ]; then
        echo "  Stopping llama-server..."
        kill $LLAMA_PID 2>/dev/null || true
        wait $LLAMA_PID 2>/dev/null || true
        unset LLAMA_PID
    fi
}

# Function to run warmup
run_warmup() {
    echo "  Running warmup ($WARMUP_PROMPTS prompts)..."
    cd "$LLAMA_WORKDIR"

    uv run vllm bench serve \
        --model Qwen/Qwen3-30B-A3B-FP8 \
        --dataset-name sharegpt \
        --num-prompts $WARMUP_PROMPTS \
        --dataset-path "$DATASET_PATH" \
        --base-url http://127.0.0.1:$SERVER_PORT \
        --max-concurrency=$MAX_CONCURRENCY \
        --request-rate $REQUEST_RATE \
        > /dev/null 2>&1 || true

    echo "  Warmup completed"
    sleep 2
}

# Function to run benchmark (multiple runs for stability)
run_benchmark() {
    local scenario=$1
    local mode=$2  # "trace" or "notrace"
    local run_num=$3  # run number (1, 2, 3, ...)

    echo "  Running vllm bench (run $run_num/$NUM_RUNS, $NUM_PROMPTS prompts)..."
    cd "$LLAMA_WORKDIR"

    uv run vllm bench serve \
        --model Qwen/Qwen3-30B-A3B-FP8 \
        --dataset-name sharegpt \
        --num-prompts $NUM_PROMPTS \
        --dataset-path "$DATASET_PATH" \
        --base-url http://127.0.0.1:$SERVER_PORT \
        --max-concurrency=$MAX_CONCURRENCY \
        --request-rate $REQUEST_RATE \
        2>&1 | tee "$OUTPUT_DIR/${scenario}_${mode}_run${run_num}.log"
}

# Function to start tracing
start_tracing() {
    local scenario=$1
    echo "  Starting cuda_sched_trace..."
    sudo "$TRACE_TOOL" > "$OUTPUT_DIR/${scenario}_trace.csv" 2> "$OUTPUT_DIR/${scenario}_trace.log" &
    TRACE_PID=$!
    sleep 2
}

# Function to stop tracing
stop_tracing() {
    if [ -n "$TRACE_PID" ]; then
        echo "  Stopping cuda_sched_trace..."
        sudo kill -SIGINT $TRACE_PID 2>/dev/null || true
        wait $TRACE_PID 2>/dev/null || true
        unset TRACE_PID
    fi
}

# Function to start CPU stress
start_cpu_stress() {
    echo "  Starting CPU stress (stress-ng)..."
    stress-ng --cpu 0 --cpu-method fft --metrics-brief &
    STRESS_PID=$!
    sleep 2
}

# Function to stop CPU stress
stop_cpu_stress() {
    if [ -n "$STRESS_PID" ]; then
        kill $STRESS_PID 2>/dev/null || true
        wait $STRESS_PID 2>/dev/null || true
        unset STRESS_PID
    fi
}

# Function to start network stress
start_network_stress() {
    echo "  Starting network stress (iperf3)..."
    iperf3 -s -p 5201 &
    IPERF_SERVER=$!
    sleep 1
    iperf3 -c 127.0.0.1 -p 5201 -t 600 -P 10 &
    IPERF_CLIENT=$!
    sleep 2
}

# Function to stop network stress
stop_network_stress() {
    kill $IPERF_CLIENT $IPERF_SERVER 2>/dev/null || true
    wait $IPERF_CLIENT 2>/dev/null || true
    wait $IPERF_SERVER 2>/dev/null || true
    unset IPERF_CLIENT IPERF_SERVER
}

# Function to start disk stress
start_disk_stress() {
    echo "  Starting disk stress (fio)..."
    fio --name=randwrite --ioengine=libaio --iodepth=32 --rw=randwrite --bs=4k --direct=1 \
        --size=1G --numjobs=4 --runtime=600 --time_based --filename=/tmp/fio_test &
    FIO_PID=$!
    sleep 2
}

# Function to stop disk stress
stop_disk_stress() {
    if [ -n "$FIO_PID" ]; then
        kill $FIO_PID 2>/dev/null || true
        wait $FIO_PID 2>/dev/null || true
        rm -f /tmp/fio_test
        unset FIO_PID
    fi
}

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    stop_tracing
    stop_cpu_stress
    stop_network_stress
    stop_disk_stress
    stop_llama_server
}

trap cleanup EXIT

# =============================================================================
# Test Scenarios
# =============================================================================

run_scenario() {
    local scenario=$1
    local with_trace=$2
    local noisy_type=$3
    local pinned=$4

    local mode="notrace"
    [ "$with_trace" = "true" ] && mode="trace"

    echo "-------------------------------------------------------------------"
    echo "Scenario: $scenario (mode: $mode) - $NUM_RUNS runs x $NUM_PROMPTS prompts"
    echo "-------------------------------------------------------------------"

    # Start llama-server
    start_llama_server "$pinned"

    # Run warmup first (without tracing, without noisy neighbors)
    run_warmup

    # Start tracing if needed
    [ "$with_trace" = "true" ] && start_tracing "$scenario"

    # Start noisy neighbors
    case "$noisy_type" in
        "cpu")
            start_cpu_stress
            ;;
        "network")
            start_network_stress
            ;;
        "disk")
            start_disk_stress
            ;;
        "heavy")
            start_cpu_stress
            start_network_stress
            start_disk_stress
            ;;
        *)
            # No noisy neighbor
            ;;
    esac

    # Run benchmark multiple times for stability
    for run in $(seq 1 $NUM_RUNS); do
        run_benchmark "$scenario" "$mode" "$run"
        sleep 3  # Brief pause between runs
    done

    # Stop noisy neighbors
    case "$noisy_type" in
        "cpu")
            stop_cpu_stress
            ;;
        "network")
            stop_network_stress
            ;;
        "disk")
            stop_disk_stress
            ;;
        "heavy")
            stop_cpu_stress
            stop_network_stress
            stop_disk_stress
            ;;
    esac

    # Stop tracing if needed
    [ "$with_trace" = "true" ] && stop_tracing

    # Stop llama-server
    stop_llama_server

    echo "Completed: $scenario ($mode)"
    echo ""
    sleep 5
}

# =============================================================================
# Main Test Execution
# =============================================================================

echo ""
echo "Phase 1: Tests WITHOUT tracing (actual performance)"
echo "==================================================================="

echo "[1/12] Baseline - No Trace"
run_scenario "baseline" "false" "none" "normal"

echo "[2/12] CPU Stress - No Trace"
run_scenario "cpu_stress" "false" "cpu" "normal"

echo "[3/12] Network Stress - No Trace"
run_scenario "network_stress" "false" "network" "normal"

echo "[4/12] Disk Stress - No Trace"
run_scenario "disk_stress" "false" "disk" "normal"

echo "[5/12] Heavy Load - No Trace"
run_scenario "heavy_load" "false" "heavy" "normal"

echo "[6/12] CPU Pinned - No Trace"
run_scenario "cpu_pinned" "false" "cpu" "pinned"

echo ""
echo "Phase 2: Tests WITH tracing (scheduler metrics)"
echo "==================================================================="

echo "[7/12] Baseline - With Trace"
run_scenario "baseline" "true" "none" "normal"

echo "[8/12] CPU Stress - With Trace"
run_scenario "cpu_stress" "true" "cpu" "normal"

echo "[9/12] Network Stress - With Trace"
run_scenario "network_stress" "true" "network" "normal"

echo "[10/12] Disk Stress - With Trace"
run_scenario "disk_stress" "true" "disk" "normal"

echo "[11/12] Heavy Load - With Trace"
run_scenario "heavy_load" "true" "heavy" "normal"

echo "[12/12] CPU Pinned - With Trace"
run_scenario "cpu_pinned" "true" "cpu" "pinned"

echo ""
echo "==================================================================="
echo "All tests completed!"
echo "Results in: $OUTPUT_DIR"
echo "==================================================================="

# Save test configuration
cat > "$OUTPUT_DIR/config.txt" << EOF
Test Date: $TEST_DATE
NUM_PROMPTS: $NUM_PROMPTS
NUM_RUNS: $NUM_RUNS
WARMUP_PROMPTS: $WARMUP_PROMPTS
REQUEST_RATE: $REQUEST_RATE
MAX_CONCURRENCY: $MAX_CONCURRENCY
SERVER_PORT: $SERVER_PORT
EOF

# Run analysis
echo ""
echo "Running analysis..."
python3 "$SCRIPT_DIR/analyze_sched_overhead.py" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/analysis.txt"

# Create symlink to latest results
ln -sfn "$OUTPUT_DIR" "$SCRIPT_DIR/results/latest"

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Symlink created: $SCRIPT_DIR/results/latest"

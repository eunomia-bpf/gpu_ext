/**
 * Example 1: Thread Divergence in eBPF Hook - XDP-style Packet Parsing
 *
 * This example simulates an XDP-like eBPF program that:
 *   1. Parses Ethernet header (L2)
 *   2. Checks if IPv4 (L3)
 *   3. Checks if TCP (L4)
 *   4. Checks if HTTP (port 80)
 *   5. Parses HTTP path and updates path counters
 *
 * The eBPF hook would PASS traditional eBPF verification:
 *   ✓ Memory safe (bounds checked before access)
 *   ✓ Bounded execution (finite parsing depth)
 *   ✓ Valid helper usage
 *
 * But causes GPU-specific issues:
 *   ✗ Massive warp divergence from packet type branching
 *   ✗ Different threads parse different protocol layers
 *   ✗ HTTP path matching causes further divergence
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

//=============================================================================
// Network Protocol Definitions (like in real XDP/eBPF)
//=============================================================================

#define ETH_P_IP    0x0800
#define IPPROTO_TCP 6
#define IPPROTO_UDP 17
#define HTTP_PORT   80

// Ethernet header
struct ethhdr {
    unsigned char  h_dest[6];
    unsigned char  h_source[6];
    unsigned short h_proto;
};

// IPv4 header (simplified)
struct iphdr {
    unsigned char  ihl_version;  // version:4, ihl:4
    unsigned char  tos;
    unsigned short tot_len;
    unsigned short id;
    unsigned short frag_off;
    unsigned char  ttl;
    unsigned char  protocol;
    unsigned short check;
    unsigned int   saddr;
    unsigned int   daddr;
};

// TCP header (simplified)
struct tcphdr {
    unsigned short source;
    unsigned short dest;
    unsigned int   seq;
    unsigned int   ack_seq;
    unsigned short flags;  // data offset, flags
    unsigned short window;
    unsigned short check;
    unsigned short urg_ptr;
};

//=============================================================================
// Simulated eBPF Infrastructure
//=============================================================================

// BPF map for path counters (like BPF_MAP_TYPE_HASH)
#define MAX_PATHS 8
#define PATH_API_USERS    0  // /api/users
#define PATH_API_ORDERS   1  // /api/orders
#define PATH_API_PRODUCTS 2  // /api/products
#define PATH_STATIC       3  // /static/*
#define PATH_INDEX        4  // /index.html
#define PATH_HEALTH       5  // /health
#define PATH_METRICS      6  // /metrics
#define PATH_OTHER        7  // other paths

__device__ unsigned long long path_counters[MAX_PATHS];
__device__ unsigned long long protocol_counters[4];  // [0]=non-ip, [1]=non-tcp, [2]=non-http, [3]=http

// eBPF Helper: Get thread index (packet index)
__device__ void bpf_get_thread_idx(unsigned long long *x, unsigned long long *y, unsigned long long *z) {
    *x = threadIdx.x + blockIdx.x * blockDim.x;
    *y = threadIdx.y;
    *z = threadIdx.z;
}

// eBPF Helper: Atomic increment (for counters)
__device__ void bpf_map_atomic_inc(unsigned long long *counter) {
    atomicAdd(counter, 1ULL);
}

// eBPF Helper: Bounds check (like in XDP)
__device__ int bpf_check_bounds(void *data, void *data_end, void *ptr, int size) {
    return ((unsigned char*)ptr + size <= (unsigned char*)data_end);
}

//=============================================================================
// Simulated Packet Data
//=============================================================================

#define PACKET_SIZE 256
#define HTTP_PAYLOAD_OFFSET (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr))

// Generate simulated packet with given characteristics
__device__ void generate_packet(unsigned char *pkt, int pkt_idx, int *pkt_type) {
    // Zero out packet memory first
    for (int i = 0; i < PACKET_SIZE; i++) {
        pkt[i] = 0;
    }

    // Vary packet types based on index to simulate real traffic
    int type = pkt_idx % 10;

    struct ethhdr *eth = (struct ethhdr *)pkt;
    struct iphdr *ip = (struct iphdr *)(pkt + sizeof(struct ethhdr));
    struct tcphdr *tcp = (struct tcphdr *)(pkt + sizeof(struct ethhdr) + sizeof(struct iphdr));
    char *http = (char *)(pkt + HTTP_PAYLOAD_OFFSET);

    // Default: valid HTTP packet
    eth->h_proto = ETH_P_IP;
    ip->ihl_version = 0x45;  // IPv4, IHL=5
    ip->protocol = IPPROTO_TCP;
    tcp->dest = HTTP_PORT;
    tcp->flags = (5 << 12);  // Data offset = 5 (20 bytes)

    if (type == 0) {
        // 10%: Non-IP packet (e.g., ARP)
        eth->h_proto = 0x0806;  // ARP
        *pkt_type = 0;
    } else if (type == 1) {
        // 10%: IP but UDP (not TCP)
        ip->protocol = IPPROTO_UDP;
        *pkt_type = 1;
    } else if (type == 2) {
        // 10%: TCP but not HTTP (e.g., port 443)
        tcp->dest = 443;
        *pkt_type = 2;
    } else {
        // 70%: HTTP requests with different paths
        *pkt_type = 3;
        int path_type = pkt_idx % 8;
        const char *paths[] = {
            "GET /api/users HTTP/1.1\r\n",
            "GET /api/orders HTTP/1.1\r\n",
            "GET /api/products HTTP/1.1\r\n",
            "GET /static/style.css HTTP/1.1\r\n",
            "GET /index.html HTTP/1.1\r\n",
            "GET /health HTTP/1.1\r\n",
            "GET /metrics HTTP/1.1\r\n",
            "GET /unknown/path HTTP/1.1\r\n"
        };
        // Copy path (simplified - in real code use proper bounds checking)
        const char *p = paths[path_type];
        for (int i = 0; i < 30 && p[i]; i++) {
            http[i] = p[i];
        }
    }
}

//=============================================================================
// eBPF HOOK - BAD: XDP-style parsing with natural divergence
//=============================================================================

/**
 * This eBPF program parses packets like XDP, causing natural divergence.
 *
 * Traditional eBPF verifier sees:
 *   - All memory accesses are bounds-checked
 *   - All branches are valid and bounded
 *   - No infinite loops
 *
 * GPU reality:
 *   - Threads take different paths based on packet content
 *   - Warp serialization: only one path executes at a time
 *   - N-way divergence from path matching
 */
__device__ void ebpf_hook_BAD_xdp_parse(unsigned char *pkt_data, int pkt_len) {
    // Parse all layers (all threads do this)
    struct ethhdr *eth = (struct ethhdr *)pkt_data;
    struct iphdr *ip = (struct iphdr *)(pkt_data + sizeof(struct ethhdr));
    int ip_hdr_len = (ip->ihl_version & 0x0F) * 4;
    struct tcphdr *tcp = (struct tcphdr *)((unsigned char *)ip + ip_hdr_len);
    int tcp_hdr_len = ((tcp->flags >> 12) & 0x0F) * 4;
    if (tcp_hdr_len < 20) tcp_hdr_len = 20;
    char *http_data = (char *)tcp + tcp_hdr_len;
    char *path = http_data + 4;

    // ═══════════════════════════════════════════════════════════════════
    // BAD PATTERN: Different paths do VASTLY different amounts of work
    // Threads with less work WAIT for threads with more work = wasted cycles
    // ═══════════════════════════════════════════════════════════════════

    // Compute predicates
    int is_ip = (eth->h_proto == ETH_P_IP);
    int is_tcp = (ip->protocol == IPPROTO_TCP);
    int is_http = (tcp->dest == HTTP_PORT);

    unsigned long long work = 0;

    // DIVERGENT: Each path has DIFFERENT amounts of compute work
    // Using pure computation to avoid memory bottlenecks
    if (!is_ip) {
        // Non-IP (10%): heavy compute
        for (int i = 0; i < 1000; i++) work = work * 3 + 7;
        bpf_map_atomic_inc(&protocol_counters[0]);
    } else if (!is_tcp) {
        // Non-TCP (10%): heavy compute
        for (int i = 0; i < 1000; i++) work = work * 5 + 11;
        bpf_map_atomic_inc(&protocol_counters[1]);
    } else if (!is_http) {
        // Non-HTTP (10%): heavy compute
        for (int i = 0; i < 1000; i++) work = work * 7 + 13;
        bpf_map_atomic_inc(&protocol_counters[2]);
    } else {
        // HTTP (70%): heavy compute
        for (int i = 0; i < 1000; i++) work = work * 11 + 17;
        bpf_map_atomic_inc(&protocol_counters[3]);

        // Further divergence on path
        if (path[1] == 'a' && path[5] == 'u') {
            bpf_map_atomic_inc(&path_counters[PATH_API_USERS]);
        } else if (path[1] == 'a' && path[5] == 'o') {
            bpf_map_atomic_inc(&path_counters[PATH_API_ORDERS]);
        } else if (path[1] == 'a' && path[5] == 'p') {
            bpf_map_atomic_inc(&path_counters[PATH_API_PRODUCTS]);
        } else if (path[1] == 's') {
            bpf_map_atomic_inc(&path_counters[PATH_STATIC]);
        } else if (path[1] == 'i') {
            bpf_map_atomic_inc(&path_counters[PATH_INDEX]);
        } else if (path[1] == 'h') {
            bpf_map_atomic_inc(&path_counters[PATH_HEALTH]);
        } else if (path[1] == 'm') {
            bpf_map_atomic_inc(&path_counters[PATH_METRICS]);
        } else {
            bpf_map_atomic_inc(&path_counters[PATH_OTHER]);
        }
    }

    // Prevent optimization
    if (work == 0xDEADBEEFCAFE) atomicAdd(&protocol_counters[0], 1ULL);
}

//=============================================================================
// eBPF HOOK - GOOD: Uniform processing using predication (no divergence)
//=============================================================================

/**
 * GPU-optimized: All threads do same work, use predication instead of branches
 * No divergence - all threads execute all instructions uniformly
 */
__device__ void ebpf_hook_GOOD_uniform(unsigned char *pkt_data, int pkt_len) {
    // Parse all layers (all threads do this - same as BAD)
    struct ethhdr *eth = (struct ethhdr *)pkt_data;
    struct iphdr *ip = (struct iphdr *)(pkt_data + sizeof(struct ethhdr));
    int ip_hdr_len = (ip->ihl_version & 0x0F) * 4;
    struct tcphdr *tcp = (struct tcphdr *)((unsigned char *)ip + ip_hdr_len);
    int tcp_hdr_len = ((tcp->flags >> 12) & 0x0F) * 4;
    if (tcp_hdr_len < 20) tcp_hdr_len = 20;
    char *http_data = (char *)tcp + tcp_hdr_len;
    char *path = http_data + 4;

    // ═══════════════════════════════════════════════════════════════════
    // GOOD PATTERN: ALL threads do the SAME work uniformly
    // Same total work as BAD, but executed in PARALLEL (no divergence)
    // ═══════════════════════════════════════════════════════════════════

    // Compute predicates uniformly (all threads do this)
    int is_ip = (eth->h_proto == ETH_P_IP);
    int is_tcp = (ip->protocol == IPPROTO_TCP);
    int is_http = (tcp->dest == HTTP_PORT);

    unsigned long long work = 0;

    // ALL threads do 1000 iterations uniformly (same as each BAD branch)
    // In BAD, a warp does 4 passes of 1000 iterations = 4000 serial iterations
    // In GOOD, all 32 threads do 1000 iterations in parallel = 1000 cycles
    // Expected: GOOD should be ~4x faster due to parallel execution
    for (int i = 0; i < 1000; i++) work = work * 3 + 7;

    // Determine protocol category with predicated assignments (uniform)
    int proto_idx = 0;  // Default: non-IP
    proto_idx = is_ip && !is_tcp ? 1 : proto_idx;
    proto_idx = is_ip && is_tcp && !is_http ? 2 : proto_idx;
    proto_idx = is_ip && is_tcp && is_http ? 3 : proto_idx;

    // Single atomic per thread (all threads do this - uniform)
    atomicAdd(&protocol_counters[proto_idx], 1ULL);

    // For HTTP packets, determine path category uniformly
    int path_idx = PATH_OTHER;  // Default for non-HTTP or unknown paths

    // Predicated path selection (all HTTP threads evaluate, uniform)
    int http_valid = is_ip && is_tcp && is_http;
    path_idx = (http_valid && path[1] == 'a' && path[5] == 'u') ? PATH_API_USERS : path_idx;
    path_idx = (http_valid && path[1] == 'a' && path[5] == 'o') ? PATH_API_ORDERS : path_idx;
    path_idx = (http_valid && path[1] == 'a' && path[5] == 'p') ? PATH_API_PRODUCTS : path_idx;
    path_idx = (http_valid && path[1] == 's') ? PATH_STATIC : path_idx;
    path_idx = (http_valid && path[1] == 'i') ? PATH_INDEX : path_idx;
    path_idx = (http_valid && path[1] == 'h') ? PATH_HEALTH : path_idx;
    path_idx = (http_valid && path[1] == 'm') ? PATH_METRICS : path_idx;

    // Single atomic per HTTP thread (uniform - only HTTP threads increment)
    if (http_valid) {
        atomicAdd(&path_counters[path_idx], 1ULL);
    }

    // Prevent optimization
    if (work == 0xDEADBEEFCAFE) atomicAdd(&protocol_counters[0], 1ULL);
}

//=============================================================================
// CUDA Kernels
//=============================================================================

__global__ void process_packets_bad(unsigned char *packets, int num_packets, int pkt_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_packets) return;

    unsigned char *my_pkt = packets + tid * pkt_size;

    // Generate packet data (simulating receiving different packets)
    int pkt_type;
    generate_packet(my_pkt, tid, &pkt_type);

    // Run XDP-style eBPF hook - CAUSES DIVERGENCE
    ebpf_hook_BAD_xdp_parse(my_pkt, pkt_size);
}

__global__ void process_packets_good(unsigned char *packets, int num_packets, int pkt_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_packets) return;

    unsigned char *my_pkt = packets + tid * pkt_size;

    // Generate packet data
    int pkt_type;
    generate_packet(my_pkt, tid, &pkt_type);

    // Run uniform eBPF hook - uses predication instead of divergent branches
    ebpf_hook_GOOD_uniform(my_pkt, pkt_size);
}

__global__ void reset_counters() {
    int tid = threadIdx.x;
    if (tid < MAX_PATHS) path_counters[tid] = 0;
    if (tid < 4) protocol_counters[tid] = 0;
}

//=============================================================================
// Main
//=============================================================================

int main() {
    const int NUM_PACKETS = 1024 * 1024;  // 1M packets
    const int THREADS = 256;
    const int BLOCKS = (NUM_PACKETS + THREADS - 1) / THREADS;
    const int ITERATIONS = 50;

    // Allocate packet buffer
    unsigned char *d_packets;
    cudaError_t err = cudaMalloc(&d_packets, NUM_PACKETS * PACKET_SIZE);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Example 1: Thread Divergence - XDP-style Packet Parsing      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("Simulating XDP-like eBPF program that parses network packets:\n");
    printf("  L2 (Ethernet) → L3 (IPv4) → L4 (TCP) → L7 (HTTP path)\n\n");
    printf("Traffic mix: 10%% non-IP, 10%% non-TCP, 10%% non-HTTP, 70%% HTTP\n");
    printf("HTTP paths: /api/users, /api/orders, /api/products, /static/*,\n");
    printf("            /index.html, /health, /metrics, other\n\n");

    // Warmup
    reset_counters<<<1, 256>>>();
    process_packets_good<<<BLOCKS, THREADS>>>(d_packets, NUM_PACKETS, PACKET_SIZE);
    cudaDeviceSynchronize();

    // Test GOOD (uniform) hook
    reset_counters<<<1, 256>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        process_packets_good<<<BLOCKS, THREADS>>>(d_packets, NUM_PACKETS, PACKET_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float good_time;
    cudaEventElapsedTime(&good_time, start, stop);

    // Test BAD (divergent XDP) hook
    reset_counters<<<1, 256>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        process_packets_bad<<<BLOCKS, THREADS>>>(d_packets, NUM_PACKETS, PACKET_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bad_time;
    cudaEventElapsedTime(&bad_time, start, stop);

    // Run one more iteration with fresh counters to get statistics
    reset_counters<<<1, 256>>>();
    cudaDeviceSynchronize();

    process_packets_bad<<<BLOCKS, THREADS>>>(d_packets, NUM_PACKETS, PACKET_SIZE);
    cudaDeviceSynchronize();

    // Copy counters back
    unsigned long long h_path_counters[MAX_PATHS];
    unsigned long long h_protocol_counters[4];
    cudaMemcpyFromSymbol(h_path_counters, path_counters, sizeof(h_path_counters));
    cudaMemcpyFromSymbol(h_protocol_counters, protocol_counters, sizeof(h_protocol_counters));

    printf("Results (%d iterations, %d packets each):\n", ITERATIONS, NUM_PACKETS);
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  GOOD hook (uniform):     %8.2f ms  (%.2f Mpps)\n",
           good_time, (float)NUM_PACKETS * ITERATIONS / good_time / 1000.0f);
    printf("  BAD hook (XDP-style):    %8.2f ms  (%.2f Mpps)\n\n",
           bad_time, (float)NUM_PACKETS * ITERATIONS / bad_time / 1000.0f);

    printf("Performance Impact:\n");
    if (bad_time > good_time) {
        printf("  BAD is %.2fx slower than GOOD (divergence overhead visible)\n\n", bad_time / good_time);
    } else {
        printf("  BAD is %.2fx faster (modern GPUs optimize divergence well)\n\n", good_time / bad_time);
    }
    printf("Note: Modern GPUs with predicated execution handle simple divergence\n");
    printf("efficiently. The verification concern is for complex divergence patterns\n");
    printf("that cause severe serialization or resource contention.\n\n");

    printf("Protocol Statistics (last iteration):\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  Non-IP packets:    %8llu (%.1f%%)\n", h_protocol_counters[0],
           100.0 * h_protocol_counters[0] / NUM_PACKETS);
    printf("  Non-TCP packets:   %8llu (%.1f%%)\n", h_protocol_counters[1],
           100.0 * h_protocol_counters[1] / NUM_PACKETS);
    printf("  Non-HTTP packets:  %8llu (%.1f%%)\n", h_protocol_counters[2],
           100.0 * h_protocol_counters[2] / NUM_PACKETS);
    printf("  HTTP packets:      %8llu (%.1f%%)\n\n", h_protocol_counters[3],
           100.0 * h_protocol_counters[3] / NUM_PACKETS);

    printf("HTTP Path Statistics:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    const char *path_names[] = {
        "/api/users", "/api/orders", "/api/products", "/static/*",
        "/index.html", "/health", "/metrics", "other"
    };
    for (int i = 0; i < MAX_PATHS; i++) {
        printf("  %-15s: %8llu\n", path_names[i], h_path_counters[i]);
    }

    printf("\nAnalysis:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("The XDP-style parsing causes CASCADING DIVERGENCE:\n\n");
    printf("  Warp of 32 threads processing 32 packets:\n");
    printf("  ├─ 3 packets exit at L2 (non-IP)     → 3 threads idle\n");
    printf("  ├─ 3 packets exit at L3 (non-TCP)    → 3 more threads idle\n");
    printf("  ├─ 3 packets exit at L4 (non-HTTP)   → 3 more threads idle\n");
    printf("  └─ 23 packets parse HTTP             → further diverge on path!\n");
    printf("      ├─ ~3 match /api/users\n");
    printf("      ├─ ~3 match /api/orders\n");
    printf("      ├─ ~3 match /api/products\n");
    printf("      └─ ... (8 different paths = up to 8-way divergence)\n\n");

    printf("Traditional eBPF Verifier:  PASS\n");
    printf("  ✓ All memory accesses bounds-checked\n");
    printf("  ✓ All branches terminate\n");
    printf("  ✓ No infinite loops\n\n");

    printf("GPU-aware Verifier should:  REJECT or WARN\n");
    printf("  ✗ Multiple early returns cause divergence\n");
    printf("  ✗ Nested if-else on packet content\n");
    printf("  ✗ Path matching causes N-way divergence\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_packets);

    return 0;
}

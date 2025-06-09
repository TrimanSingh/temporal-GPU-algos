#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

// --- Definitions ---
using vertex_t = int;
using time_temporal = int;

const time_temporal INF_TIME = std::numeric_limits<time_temporal>::max();

struct TemporalEdge {
    vertex_t src, dst;
    time_temporal time;
};

struct EdgeData {
    std::vector<TemporalEdge> edges;
    vertex_t max_vertex_id;
};

// --- CSV Loader  ---
EdgeData load_temporal_edges_from_csv(const std::string& filename) {
    std::vector<TemporalEdge> loaded_edges;
    vertex_t current_max_id = -1;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        vertex_t u_val, v_val;
        time_temporal t_val;
        // Parse CSV columns
        std::getline(ss, token, ','); u_val = std::stoi(token);
        std::getline(ss, token, ','); v_val = std::stoi(token);
        std::getline(ss, token, ','); /* skipped weight */
        std::getline(ss, token, ','); t_val = std::stoi(token);

        loaded_edges.push_back({u_val, v_val, t_val});
        current_max_id = std::max(current_max_id, std::max(u_val, v_val));
    }

    return {loaded_edges, current_max_id};
}

// --- CSR + Timestamp builder ---
void build_csr(
    const thrust::host_vector<TemporalEdge>& edges,
    int num_vertices,
    thrust::host_vector<int>& h_idx,
    thrust::host_vector<int>& h_adj,
    thrust::host_vector<time_temporal>& h_adj_t)
{
    int M = edges.size();
    h_idx.assign(num_vertices+1, 0);
    for (auto &e: edges) {
        ++h_idx[e.src + 1];
    }
    for (int i = 1; i <= num_vertices; ++i) {
        h_idx[i] += h_idx[i-1];
    }

    h_adj.assign(M, 0);
    h_adj_t.assign(M, 0);

    thrust::host_vector<int> pos = h_idx;
    for (auto &e: edges) {
        int p = pos[e.src]++;
        h_adj[p] = e.dst;
        h_adj_t[p] = e.time;
    }

    // Sort each adjacency list by timestamp
    for (int u = 0; u < num_vertices; ++u) {
        int start = h_idx[u], end = h_idx[u+1];
        std::vector<std::pair<time_temporal,int>> tmp;
        tmp.reserve(end-start);
        for (int i = start; i < end; ++i) {
            tmp.emplace_back(h_adj_t[i], h_adj[i]);
        }
        std::sort(tmp.begin(), tmp.end());
        for (int i = start; i < end; ++i) {
            h_adj_t[i] = tmp[i-start].first;
            h_adj[i]   = tmp[i-start].second;
        }
    }
}

// --- GPU-based binary search for the closing edge (u, w) after v-w time ---
__device__ bool binary_search_closing_edge(
    const int* adj, const time_temporal* adj_t,
    int lo, int hi, int w, time_temporal t_vw, time_temporal delta)
{
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int dst = adj[mid];
        if (dst < w || (dst == w && adj_t[mid] <= t_vw))
            lo = mid + 1;
        else
            hi = mid;
    }
    return (lo < hi && adj[lo] == w && (adj_t[lo] - t_vw <= delta));
}

// --- Kernel: temporal triangle counting ---
__global__ void count_triangles_kernel(
    const int* edge_u, const int* edge_v, const time_temporal* edge_t, int M,
    const int* adj, const int* idx, const time_temporal* adj_t,
    time_temporal delta, unsigned long long* result)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= M) return;

    int u = edge_u[e];
    int v = edge_v[e];
    time_temporal t_uv = edge_t[e];

    int start = idx[v], end = idx[v+1];
    for (int i = start; i < end; ++i) {
        int w = adj[i];
        time_temporal t_vw = adj_t[i];
        if (t_vw <= t_uv || t_vw - t_uv > delta) continue;
        if (binary_search_closing_edge(adj, adj_t, idx[u], idx[u+1], w, t_vw, delta)) {
            atomicAdd(result, 1ULL);
        }
    }
}

// --- Host wrapper ---
unsigned long long count_triangles(
    const thrust::host_vector<TemporalEdge>& h_edges,
    int num_vertices, time_temporal delta)
{
    int M = h_edges.size();
    // Build CSR
    thrust::host_vector<int> h_idx, h_adj;
    thrust::host_vector<time_temporal> h_adj_t;
    build_csr(h_edges, num_vertices, h_idx, h_adj, h_adj_t);

    // Copy edges and CSR to device
    thrust::host_vector<int> h_eu(M), h_ev(M);
    thrust::host_vector<time_temporal> h_et(M);
    for (int i = 0; i < M; ++i) {
        h_eu[i] = h_edges[i].src;
        h_ev[i] = h_edges[i].dst;
        h_et[i] = h_edges[i].time;
    }

    thrust::device_vector<int> d_eu = h_eu, d_ev = h_ev;
    thrust::device_vector<time_temporal> d_et = h_et;
    thrust::device_vector<int> d_idx = h_idx, d_adj = h_adj;
    thrust::device_vector<time_temporal> d_adj_t = h_adj_t;

    thrust::device_vector<unsigned long long> d_count(1, 0ULL);

    int block = 128;
    int grid = (M + block - 1) / block;
    count_triangles_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_eu.data()),
        thrust::raw_pointer_cast(d_ev.data()),
        thrust::raw_pointer_cast(d_et.data()), M,
        thrust::raw_pointer_cast(d_adj.data()),
        thrust::raw_pointer_cast(d_idx.data()),
        thrust::raw_pointer_cast(d_adj_t.data()),
        delta,
        thrust::raw_pointer_cast(d_count.data())
    );
    cudaDeviceSynchronize();

    return d_count[0];
}

// --- main() ---
int main() {
    EdgeData ed = load_temporal_edges_from_csv("data/small_temporal_graph.csv");
    if (ed.max_vertex_id < 0) {
        std::cerr << "Error: No edges loaded.\n"; return 1;
    }
    auto& H = ed.edges;
    int n = ed.max_vertex_id + 1;
    time_temporal delta = 2;  // example temporal window

    unsigned long long triangles = count_triangles(H, n, delta);
    std::cout << "Temporal triangles (within delta=" << delta << "): " << triangles << "\n";
    return 0;
}

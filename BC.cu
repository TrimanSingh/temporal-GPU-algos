#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

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

EdgeData load_temporal_edges_from_csv(const std::string& filename) {
    std::vector<TemporalEdge> loaded_edges;
    vertex_t current_max_id = -1; // Initialize to -1, so if no edges, num_vertices becomes 0
    std::ifstream file(filename);
    std::string line;

    std::cout << "[Info] Attempting to open file: " << filename << std::endl;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {loaded_edges, -1}; // Return empty edges and -1
    }
    std::cout << "[Info] Successfully opened file: " << filename << std::endl;

    int line_number = 0; // For error reporting
    while (std::getline(file, line)) {
        line_number++;
        std::istringstream ss(line);
        std::string token;
        vertex_t u_val, v_val;
        time_temporal t_val;

        try {
            if (!std::getline(ss, token, ',')) throw std::runtime_error("Missing u");
            u_val = std::stoi(token);

            if (!std::getline(ss, token, ',')) throw std::runtime_error("Missing v");
            v_val = std::stoi(token);

            // Skip weight column
            if (!std::getline(ss, token, ',')) throw std::runtime_error("Missing w (weight/ignored)");

            if (!std::getline(ss, token, ',')) throw std::runtime_error("Missing t");
            t_val = std::stoi(token);

            loaded_edges.push_back({u_val, v_val, t_val});
            if (u_val > current_max_id) current_max_id = u_val;
            if (v_val > current_max_id) current_max_id = v_val;

            if (loaded_edges.size() <= 5 || (u_val == 0 && loaded_edges.size() <= 200 && loaded_edges.size() % 20 == 0) ) { // Print first 5, and some from source 0
                 std::cout << "[CSV Load] Edge: (" << u_val << " -> " << v_val << " @ " << t_val << ")" << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Warning: Malformed line " << line_number << " ('" << line << "'). Error: " << e.what() << ". Skipping." << std::endl;
            continue;
        }
    }

    file.close();
    std::cout << "[Info] Finished processing file. Loaded " << loaded_edges.size() << " edges. Max vertex ID found: " << current_max_id << std::endl;
    
    return {loaded_edges, current_max_id};
}

// GPU kernel to relax one edge in Brandes forward pass
__global__
void temporal_relax_edges(const TemporalEdge* edges, int num_edges,
                          time_temporal* d_sigma, int* d_dist,
                          int* d_pred_count, bool* d_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    auto e = edges[idx];
    if (d_sigma[e.src] != INF_TIME && d_dist[e.src] + 1 < d_dist[e.dst] &&
        /* temporal constraint: e.time >= d_sigma[e.src] */ e.time >= d_sigma[e.src]) {
        d_dist[e.dst] = d_dist[e.src] + 1;
        d_sigma[e.dst] = e.time;
        d_pred_count[e.dst] = d_pred_count[e.src];
        *d_active = true;
    }
    else if (d_sigma[e.src] != INF_TIME && d_dist[e.src] + 1 == d_dist[e.dst] &&
             e.time >= d_sigma[e.src]) {
        atomicAdd(&d_pred_count[e.dst], d_pred_count[e.src]);
    }
}

// GPU kernel for dependency accumulation (backwards)
__global__
void temporal_dependency(const TemporalEdge* edges, int num_edges,
                         int* d_dist, double* d_delta,
                         double* d_bc, int* d_pred_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    auto e = edges[idx];
    if (d_dist[e.dst] == d_dist[e.src] + 1) {
        double contrib = (d_pred_count[e.src] / (double)d_pred_count[e.dst]) * (1.0 + d_delta[e.dst]);
        atomicAdd(&d_delta[e.src], contrib);
    }
    if (e.dst != e.src)
        atomicAdd(&d_bc[e.dst], d_delta[e.dst]);
}

void temporal_betweenness(thrust::device_vector<TemporalEdge>& d_edges,
                          int num_vertices, thrust::device_vector<double>& d_bc) {
    int E = d_edges.size();
    const int BS = 256, BG = (E + BS - 1) / BS;

    // Sort by time so edges respect temporal ordering
    thrust::sort(d_edges.begin(), d_edges.end(),
                 [] __device__ (auto &a, auto &b) {
                     return a.time < b.time;
                 });

    thrust::device_vector<int> d_dist(num_vertices);
    thrust::device_vector<time_temporal> d_sigma(num_vertices);
    thrust::device_vector<int> d_pred_count(num_vertices);
    thrust::device_vector<double> d_delta(num_vertices);
    thrust::device_vector<bool> d_active(1);

    d_bc.assign(num_vertices, 0.0);

    for (int s = 0; s < num_vertices; ++s) {
        thrust::fill(d_dist.begin(), d_dist.end(), INT_MAX);
        thrust::fill(d_sigma.begin(), d_sigma.end(), INF_TIME);
        thrust::fill(d_pred_count.begin(), d_pred_count.end(), 0);
        thrust::fill(d_delta.begin(), d_delta.end(), 0.0);
        
        d_dist[s] = 0;
        d_sigma[s] = 0;             // source time = 0
        d_pred_count[s] = 1;

        bool h_active;
        do {
            h_active = false;
            cudaMemcpy(thrust::raw_pointer_cast(d_active.data()), &h_active, sizeof(bool), cudaMemcpyHostToDevice);

            temporal_relax_edges<<<BG,BS>>>(
                thrust::raw_pointer_cast(d_edges.data()), E,
                thrust::raw_pointer_cast(d_sigma.data()),
                thrust::raw_pointer_cast(d_dist.data()),
                thrust::raw_pointer_cast(d_pred_count.data()),
                thrust::raw_pointer_cast(d_active.data())
            );
            cudaMemcpy(&h_active, thrust::raw_pointer_cast(d_active.data()), sizeof(bool), cudaMemcpyDeviceToHost);
        } while (h_active);

        // Run dependency accumulation in reverse temporal order (approximate)
        temporal_dependency<<<BG,BS>>>(
            thrust::raw_pointer_cast(d_edges.data()), E,
            thrust::raw_pointer_cast(d_dist.data()),
            thrust::raw_pointer_cast(d_delta.data()),
            thrust::raw_pointer_cast(d_bc.data()),
            thrust::raw_pointer_cast(d_pred_count.data())
        );
        cudaDeviceSynchronize();
    }
}

int main() {
    auto edge_info = load_temporal_edges_from_csv("../data/small_temporal_graph.csv");
    int N = edge_info.max_vertex_id + 1;
    thrust::host_vector<TemporalEdge> h_edges = edge_info.edges;
    thrust::device_vector<TemporalEdge> d_edges = h_edges;
    thrust::device_vector<double> d_bc(N);

    temporal_betweenness(d_edges, N, d_bc);

    auto h_bc = d_bc;
    for (int v = 0; v < N; ++v)
        std::cout << "BC[" << v << "] = " << h_bc[v] << "\n";
    return 0;
}

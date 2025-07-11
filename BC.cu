#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

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

// GPU kernel to relax edges for multiple sources simultaneously
__global__
void temporal_relax_edges_multi_source(const TemporalEdge* edges, int num_edges,
                                       time_temporal* d_sigma, int* d_dist,
                                       int* d_pred_count, bool* d_active,
                                       int num_vertices, int num_sources, int* source_list) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges * num_sources) return;
    
    int edge_idx = idx % num_edges;
    int source_idx = idx / num_edges;
    
    if (source_idx >= num_sources) return;
    
    int source_offset = source_idx * num_vertices;
    auto e = edges[edge_idx];
    
    int src_dist_idx = source_offset + e.src;
    int dst_dist_idx = source_offset + e.dst;
    
    if (d_sigma[src_dist_idx] != INF_TIME && 
        d_dist[src_dist_idx] + 1 < d_dist[dst_dist_idx] &&
        e.time >= d_sigma[src_dist_idx]) {
        
        d_dist[dst_dist_idx] = d_dist[src_dist_idx] + 1;
        d_sigma[dst_dist_idx] = e.time;
        d_pred_count[dst_dist_idx] = d_pred_count[src_dist_idx];
        d_active[source_idx] = true;
    }
    else if (d_sigma[src_dist_idx] != INF_TIME && 
             d_dist[src_dist_idx] + 1 == d_dist[dst_dist_idx] &&
             e.time >= d_sigma[src_dist_idx]) {
        atomicAdd(&d_pred_count[dst_dist_idx], d_pred_count[src_dist_idx]);
    }
}

// GPU kernel for dependency accumulation for multiple sources
__global__
void temporal_dependency_multi_source(const TemporalEdge* edges, int num_edges,
                                     int* d_dist, double* d_delta,
                                     double* d_bc, int* d_pred_count,
                                     int num_vertices, int num_sources, int* source_list) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges * num_sources) return;
    
    int edge_idx = idx % num_edges;
    int source_idx = idx / num_edges;
    
    if (source_idx >= num_sources) return;
    
    int source_offset = source_idx * num_vertices;
    auto e = edges[edge_idx];
    
    int src_idx = source_offset + e.src;
    int dst_idx = source_offset + e.dst;
    
    if (d_dist[dst_idx] == d_dist[src_idx] + 1 && d_pred_count[dst_idx] > 0) {
        double contrib = (d_pred_count[src_idx] / (double)d_pred_count[dst_idx]) * (1.0 + d_delta[dst_idx]);
        atomicAdd(&d_delta[src_idx], contrib);
    }
    
    // Accumulate to global BC (need to be careful about race conditions)
    if (e.dst != source_list[source_idx]) {
        atomicAdd(&d_bc[e.dst], d_delta[dst_idx]);
    }
}

void temporal_betweenness_parallel(thrust::device_vector<TemporalEdge>& d_edges,
                                   int num_vertices, thrust::device_vector<double>& d_bc) {
    int E = d_edges.size();
    
    std::cout << "[Info] Sorting " << E << " edges by time..." << std::endl;
    // Sort by time so edges respect temporal ordering
    thrust::sort(d_edges.begin(), d_edges.end(),
                 [] __device__ (auto &a, auto &b) {
                     return a.time < b.time;
                 });

    d_bc.assign(num_vertices, 0.0);

    std::cout << "[Info] Starting parallel betweenness computation for " << num_vertices << " vertices..." << std::endl;
    
    // Process vertices in batches to manage memory usage
    int max_vertices_to_process = std::min(num_vertices, 100);
    int batch_size = 8; // Process 8 sources simultaneously
    std::cout << "[Info] Processing first " << max_vertices_to_process << " vertices in batches of " << batch_size << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int batch_start = 0; batch_start < max_vertices_to_process; batch_start += batch_size) {
        int current_batch_size = std::min(batch_size, max_vertices_to_process - batch_start);
        
        std::cout << "[Progress] Processing batch " << (batch_start/batch_size + 1) 
                  << "/" << ((max_vertices_to_process + batch_size - 1)/batch_size)
                  << " (vertices " << batch_start << "-" << (batch_start + current_batch_size - 1) << ")" << std::endl;
        
        // Allocate memory for multiple sources
        thrust::device_vector<int> d_dist(num_vertices * current_batch_size);
        thrust::device_vector<time_temporal> d_sigma(num_vertices * current_batch_size);
        thrust::device_vector<int> d_pred_count(num_vertices * current_batch_size);
        thrust::device_vector<double> d_delta(num_vertices * current_batch_size);
        thrust::device_vector<bool> d_active(current_batch_size);
        thrust::device_vector<int> d_source_list(current_batch_size);
        
        // Initialize source list
        thrust::host_vector<int> h_source_list(current_batch_size);
        for (int i = 0; i < current_batch_size; ++i) {
            h_source_list[i] = batch_start + i;
        }
        d_source_list = h_source_list;
        
        // Initialize arrays for all sources in batch
        thrust::fill(d_dist.begin(), d_dist.end(), INT_MAX);
        thrust::fill(d_sigma.begin(), d_sigma.end(), INF_TIME);
        thrust::fill(d_pred_count.begin(), d_pred_count.end(), 0);
        thrust::fill(d_delta.begin(), d_delta.end(), 0.0);
        
        // Set initial values for each source
        for (int i = 0; i < current_batch_size; ++i) {
            int source = batch_start + i;
            int offset = i * num_vertices;
            d_dist[offset + source] = 0;
            d_sigma[offset + source] = 0;
            d_pred_count[offset + source] = 1;
        }
        
        // Calculate grid size for parallel processing
        const int BS = 256;
        const int BG = (E * current_batch_size + BS - 1) / BS;
        
        // Forward pass - relaxation
        bool h_any_active;
        int iteration = 0;
        int max_iterations = 50;
        
        do {
            // Reset active flags
            thrust::fill(d_active.begin(), d_active.end(), false);
            
            temporal_relax_edges_multi_source<<<BG, BS>>>(
                thrust::raw_pointer_cast(d_edges.data()), E,
                thrust::raw_pointer_cast(d_sigma.data()),
                thrust::raw_pointer_cast(d_dist.data()),
                thrust::raw_pointer_cast(d_pred_count.data()),
                thrust::raw_pointer_cast(d_active.data()),
                num_vertices, current_batch_size,
                thrust::raw_pointer_cast(d_source_list.data())
            );
            
            // Check if any source is still active
            h_any_active = thrust::reduce(d_active.begin(), d_active.end(), false, thrust::logical_or<bool>());
            
            iteration++;
            if (iteration >= max_iterations) {
                std::cout << "[Warning] Max iterations (" << max_iterations << ") reached for batch" << std::endl;
                break;
            }
        } while (h_any_active);
        
        std::cout << "[Info] Forward pass completed in " << iteration << " iterations" << std::endl;
        
        // Backward pass - dependency accumulation
        temporal_dependency_multi_source<<<BG, BS>>>(
            thrust::raw_pointer_cast(d_edges.data()), E,
            thrust::raw_pointer_cast(d_dist.data()),
            thrust::raw_pointer_cast(d_delta.data()),
            thrust::raw_pointer_cast(d_bc.data()),
            thrust::raw_pointer_cast(d_pred_count.data()),
            num_vertices, current_batch_size,
            thrust::raw_pointer_cast(d_source_list.data())
        );
        
        cudaDeviceSynchronize();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "[Info] Completed parallel betweenness computation in " << total_elapsed << " seconds" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [gpu_device_id]" << std::endl;
        std::cerr << "  input_file: Path to CSV file containing temporal edges" << std::endl;
        std::cerr << "  gpu_device_id: Optional GPU device ID (default: 0)" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    int gpu_device = 0;  // Default GPU device

    // Parse optional GPU device argument
    if (argc >= 3) {
        try {
            gpu_device = std::stoi(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid GPU device ID '" << argv[2] << "'. Using default GPU 0." << std::endl;
            gpu_device = 0;
        }
    }

    // Set GPU device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (gpu_device >= device_count || gpu_device < 0) {
        std::cerr << "Error: GPU device " << gpu_device << " not available. Available devices: 0-" << (device_count-1) << std::endl;
        return 1;
    }
    
    cudaSetDevice(gpu_device);
    std::cout << "[Info] Using GPU device " << gpu_device << std::endl;

    // Load temporal edges from specified file
    auto edge_info = load_temporal_edges_from_csv(input_file);
    if (edge_info.max_vertex_id == -1) {
        std::cerr << "Error: Failed to load edges from file: " << input_file << std::endl;
        return 1;
    }

    int N = edge_info.max_vertex_id + 1;
    thrust::host_vector<TemporalEdge> h_edges = edge_info.edges;
    thrust::device_vector<TemporalEdge> d_edges = h_edges;
    thrust::device_vector<double> d_bc(N);

    std::cout << "[Info] Computing temporal betweenness centrality for " << N << " vertices and " << h_edges.size() << " edges..." << std::endl;

    temporal_betweenness_parallel(d_edges, N, d_bc);


    double norm_factor = static_cast<double>((N - 1) * (N - 2));  // For directed graphs

    thrust::transform(
        d_bc.begin(), d_bc.end(), d_bc.begin(),
        [=] __device__ (double val) {
            return val / norm_factor;
        }
    );
    auto h_bc = d_bc;

    // for (int v = 0; v < N; ++v)
    //     // std::cout << "BC[" << v << "] = " << h_bc[v] << "\n";
    // return 0;
}
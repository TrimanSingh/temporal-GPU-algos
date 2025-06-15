#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector> 
#include <string>
#include <algorithm> 

using vertex_t = int;
using time_temporal = int;

struct TemporalEdge {
    vertex_t src;
    vertex_t dst;
    time_temporal time;
};

struct EdgeData {
    std::vector<TemporalEdge> edges;
    vertex_t max_vertex_id;
};

const time_temporal INF_TIME = std::numeric_limits<time_temporal>::max();


__global__ void temporal_bfs_kernel(
    const TemporalEdge* edges, int num_edges,
    int num_vertices,               // Added num_vertices for bounds checking
    const time_temporal* sigma_in,  // Represents d_sigma_current
    time_temporal* sigma_out,       // Represents d_sigma_next (initialized from d_sigma_current)
    const int* dist_in,             // Represents d_dist_current
    int* dist_out,                  // Represents d_dist_next (initialized from d_dist_current)
    int* pred_out,                  // Represents d_pred_next (initialized from d_pred_current)
    bool* changed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    TemporalEdge e = edges[idx];
    time_temporal t = e.time;
    vertex_t src_node = e.src;
    vertex_t dst_node = e.dst;

    // Bounds checking for src_node and dst_node
    if (src_node < 0 || src_node >= num_vertices || dst_node < 0 || dst_node >= num_vertices) {
        return;
    }

    // Ensure source is reachable in time for this edge's departure
    if (sigma_in[src_node] != INF_TIME && sigma_in[src_node] <= t) {
        time_temporal old_sigma_at_dst = atomicMin(&sigma_out[dst_node], t);
        if (t < old_sigma_at_dst) {
            dist_out[dst_node] = dist_in[src_node] + 1;
            pred_out[dst_node] = src_node;
            *changed = true; 
        }
    }
}

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


// Modified temporal_bfs function
void temporal_bfs(
    thrust::device_vector<TemporalEdge>& d_edges,
    int num_vertices, vertex_t source, time_temporal t0,
    // Output parameters
    thrust::device_vector<time_temporal>& result_sigma,
    thrust::device_vector<int>& result_dist,
    thrust::device_vector<int>& result_pred
) {
    int num_edges = d_edges.size();
    std::cout << "[temporal_bfs] Called with num_edges: " << num_edges
              << ", num_vertices: " << num_vertices
              << ", source: " << source
              << ", t0: " << t0 << std::endl;

    if (num_vertices <= 0) {
        std::cout << "[temporal_bfs] num_vertices is " << num_vertices << ". No BFS performed." << std::endl;
        return;
    }
     if (source < 0 || source >= num_vertices) {
         std::cerr << "[temporal_bfs] Error: Source vertex " << source << " is out of bounds for num_vertices " << num_vertices << "." << std::endl;
    }

    // Computation vectors
    thrust::device_vector<time_temporal> d_sigma_current(num_vertices, INF_TIME);
    thrust::device_vector<time_temporal> d_sigma_next(num_vertices, INF_TIME);
    thrust::device_vector<int> d_dist_current(num_vertices, std::numeric_limits<int>::max());
    thrust::device_vector<int> d_dist_next(num_vertices, std::numeric_limits<int>::max());
    thrust::device_vector<int> d_pred_in(num_vertices, -1); // Assuming -1 is an invalid predecessor
    thrust::device_vector<int> d_pred_out(num_vertices, -1);

    // Initialize source conditions
    if (source >= 0 && source < num_vertices) {
        d_sigma_current[source] = t0;
        d_dist_current[source] = 0;
        // d_pred_in[source] = source; 
    }
    // else: source is out of bounds, handled by the check above.

    bool h_changed = false; // Initialize to false for the first iteration check
    bool *d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    const int BLOCK_SIZE = 256;
    int num_blocks = (num_edges > 0) ? (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE : 0;
    
    std::cout << "[temporal_bfs] Starting iterations. num_blocks: " << num_blocks << ", BLOCK_SIZE: " << BLOCK_SIZE << std::endl;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

    int iteration = 0;

    const int MAX_ITERATIONS = (num_vertices > 0) ? (num_vertices + 5) : 10; 

    do {
        h_changed = false; // Reset for current iteration
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        d_sigma_next = d_sigma_current;
        d_dist_next = d_dist_current;
        d_pred_out = d_pred_in; // Carry over predecessors

        if (num_edges > 0) { // Only launch kernel if there are edges
            temporal_bfs_kernel<<<num_blocks, BLOCK_SIZE>>>(
                thrust::raw_pointer_cast(d_edges.data()), num_edges,
                num_vertices, // Pass num_vertices to the kernel
                thrust::raw_pointer_cast(d_sigma_current.data()),
                thrust::raw_pointer_cast(d_sigma_next.data()),
                thrust::raw_pointer_cast(d_dist_current.data()),
                thrust::raw_pointer_cast(d_dist_next.data()),
                thrust::raw_pointer_cast(d_pred_out.data()),
                d_changed
            );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "[temporal_bfs] CUDA error after kernel launch (Iteration " << iteration << "): " << cudaGetErrorString(err) << std::endl;
                cudaFree(d_changed);
                return; // Critical error, stop BFS
            }
            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
            if (iteration == 0) { // Specifically check after the first kernel execution
                std::cout << "[temporal_bfs] After 1st kernel call & copy, h_changed: " << (h_changed ? "true" : "false") << std::endl;
            }
        } else {
            // No edges, so h_changed remains false, loop will terminate.
             if (iteration == 0) { // If no edges, h_changed will be false
                std::cout << "[temporal_bfs] After 1st iteration (no edges), h_changed: false" << std::endl;
            }
        }


        std::swap(d_sigma_current, d_sigma_next);
        std::swap(d_dist_current, d_dist_next);
        std::swap(d_pred_in, d_pred_out);
        
        std::cout << "[temporal_bfs] Iteration: " << iteration << ", h_changed: " << (h_changed ? "true":"false") << std::endl;
        iteration++;
        
        if (iteration >= MAX_ITERATIONS && h_changed) {
            std::cout << "[temporal_bfs] Warning: Exceeded MAX_ITERATIONS (" << MAX_ITERATIONS << ") but h_changed is still true. Breaking loop." << std::endl;
            break;
        }
    } while (h_changed && num_edges > 0); 

    cudaFree(d_changed);

    // === Stop GPU timing ===
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "[Timing] GPU BFS Time: " << milliseconds << " ms" << std::endl;
cudaEventDestroy(start);
cudaEventDestroy(stop);


    std::cout << "[temporal_bfs] Finished iterations. Total iterations performed: " << iteration << std::endl;

    result_sigma = d_sigma_current;
    result_dist = d_dist_current;
    result_pred = d_pred_in; 
}


int main() {
    std::cout << "[Info] Main function started." << std::endl;

    EdgeData edge_info = load_temporal_edges_from_csv("../data/sx-stackoverflowC.csv");
    
    if (edge_info.max_vertex_id == -1 && edge_info.edges.empty()) {
        std::cerr << "Error: No valid edges loaded or file could not be processed correctly. Exiting." << std::endl;
        return 1;
    }

    std::vector<TemporalEdge> h_edges = edge_info.edges;
    int num_vertices = (edge_info.max_vertex_id == -1) ? 0 : edge_info.max_vertex_id + 1;

    std::cout << "[Info] Determined num_vertices: " << num_vertices << std::endl;
    std::cout << "[Info] Number of host edges loaded: " << h_edges.size() << std::endl;

    if (num_vertices == 0 && !h_edges.empty()) {
         std::cerr << "Warning: num_vertices is 0 (max_vertex_id was -1), but edges were loaded. This indicates an issue with max_vertex_id detection if edges are non-empty. Forcing num_vertices based on edges if this happens is not done here." << std::endl;
    }
     if (num_vertices > 0 && h_edges.empty()) {
        std::cout << "[Info] Graph has " << num_vertices << " vertices but no edges." << std::endl;
    }
    if (num_vertices == 0 && h_edges.empty()){
        std::cout << "[Info] Graph is empty (no vertices, no edges)." << std::endl;
    }


    thrust::device_vector<TemporalEdge> d_edges = h_edges;
    // std::cout << "[Info] Number of device edges: " << d_edges.size() << std::endl; // d_edges.size() is same as h_edges.size()

    vertex_t source = 50; // Default source
    time_temporal t0 = 0;   // Default start time

    if (num_vertices > 0 && (source < 0 || source >= num_vertices)) {
        std::cerr << "Error: Source vertex " << source << " is out of bounds for num_vertices " << num_vertices << ". Setting source to 0 if possible, or exiting if num_vertices is 0." << std::endl;
        if (num_vertices == 0) return 1; // empty graph
        source = 0; 
        std::cout << "[Info] Fallback: Source vertex set to 0." << std::endl;
    }
    
    if (num_vertices == 0) { // If graph is empty after all checks
        std::cout << "[Info] Graph is effectively empty (num_vertices = 0). BFS will not run meaningfully." << std::endl;
    }

    thrust::device_vector<time_temporal> d_final_sigma(num_vertices); // Size 0 if num_vertices is 0
    thrust::device_vector<int> d_final_dist(num_vertices);          // Size 0 if num_vertices is 0
    thrust::device_vector<int> d_final_pred(num_vertices);

    std::cout << "[Info] Calling temporal_bfs..." << std::endl;
    temporal_bfs(d_edges, num_vertices, source, t0, d_final_sigma, d_final_dist, d_final_pred);
    std::cout << "[Info] temporal_bfs call finished." << std::endl;

    thrust::host_vector<time_temporal> h_sigma_values = d_final_sigma;
    thrust::host_vector<int> h_dist_values = d_final_dist;
    thrust::host_vector<int> h_pred_values = d_final_pred;

    std::cout << "[Info] Preparing to print results. Number of vertices to print: " << num_vertices << std::endl;
    std::cout << "Temporal BFS Results (source: " << source << ", t0: " << t0 << "):" << std::endl;
    
    if (num_vertices == 0) {
        std::cout << "Graph is empty. No results to display." << std::endl;
    } else if (h_sigma_values.empty() && num_vertices > 0) {
        std::cout << "Warning: Results vectors are empty, but num_vertices > 0. Something went wrong." << std::endl;
    }


    for (int i = 0; i < num_vertices; ++i) {

        if (h_sigma_values[i] == INF_TIME) {
            // std::cout << "Earliest Arrival Time = Unreachable";
        } else {
            std::cout << "Vertex " << i << ": "<< "Earliest Arrival Time = " << h_sigma_values[i];
        }
        
        if (h_dist_values[i] == std::numeric_limits<int>::max()) {
            // std::cout << ", Distance = Unreachable" << std::endl;
        } else {
            std::cout << ", Distance = " << h_dist_values[i] << std::endl;
        }
    }

    // auto print_path = [&](int target) {
    //     std::vector<int> path;
    //     for (int v = target; v != -1 && v != source; v = h_pred_values[v]) {
    //         if (v < 0 || v >= num_vertices) break; // defensive
    //         path.push_back(v);
    //     }
    //     if (target == source || !path.empty()) {
    //         path.push_back(source);
    //         std::reverse(path.begin(), path.end());
    //         std::cout << "Path to Vertex " << target << ": ";
    //         for (size_t i = 0; i < path.size(); ++i) {
    //             std::cout << path[i];
    //             if (i + 1 < path.size()) std::cout << " -> ";
    //         }
    //         std::cout << std::endl;
    //     } else {
    //         // std::cout << "No path to Vertex " << target << " (unreachable)" << std::endl;
    //     }
    // };

    // std::cout << "\nReconstructed Paths:" << std::endl;
    // for (int i = 0; i < num_vertices; ++i) {
    //     print_path(i);
    // }


    return 0;
}
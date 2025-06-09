#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <iostream>
#include <climits> 
#include <fstream>  
#include <sstream>  
#include <vector>  

struct Edge {
    int dest;
    int timestamp;
    int weight;
};

struct BySourceThenTime {
    __host__ __device__
    bool operator()(const int4& a, const int4& b) const {
        return (a.x < b.x) || (a.x == b.x && a.z < b.z);
    }
};



void device_convert_to_csr(
    thrust::device_vector<int4>& input_edges,
    int num_nodes,
    thrust::device_vector<int>& d_row_offsets,
    thrust::device_vector<Edge>& d_edges
) {
    int num_edges = input_edges.size();

    // Step 1: Sort input by source (u) then timestamp (t)
    thrust::sort(input_edges.begin(), input_edges.end(), BySourceThenTime());

    // Step 2: Build Edge array
    d_edges.resize(num_edges);
    thrust::transform(input_edges.begin(), input_edges.end(), d_edges.begin(),
        [] __device__ (int4 e) {
            return Edge{e.y, e.z, e.w};  // dest = v, timestamp = t, weight = w
        }
    );

    // Step 3: Build row_offsets
    d_row_offsets.resize(num_nodes + 1, 0);
    thrust::device_vector<int> src_nodes(num_edges);

    // Extract u from each int4
    thrust::transform(input_edges.begin(), input_edges.end(), src_nodes.begin(),
        [] __device__ (int4 e) { return e.x; }
    );

    // Count edges per node
    thrust::device_vector<int> keys(num_edges);
    thrust::device_vector<int> counts(num_edges);

    auto new_end = thrust::reduce_by_key(
        src_nodes.begin(), src_nodes.end(),
        thrust::make_constant_iterator(1),
        keys.begin(),
        counts.begin()
    );

    int unique_keys = new_end.first - keys.begin();

    thrust::device_vector<int> temp_offsets(num_nodes + 1, 0);
    thrust::scatter(
        counts.begin(), counts.begin() + unique_keys,
        keys.begin(),
        temp_offsets.begin() + 1
    );

    // Inclusive scan to generate row_offsets
    thrust::inclusive_scan(temp_offsets.begin(), temp_offsets.end(), d_row_offsets.begin());
}

__global__ void temporal_bfs_kernel(
    int* row_offsets, Edge* edges,
    int* frontier, int* next_frontier,
    int* visited_time, int* next_frontier_size,
    int frontier_size, int current_time,
    int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];
    if (u == -1) return; // Should not happen if frontier is compacted correctly, but good check
    
    int start = row_offsets[u];
    int end = row_offsets[u + 1];

for (int i = start; i < end; ++i) {
    Edge e = edges[i];
    if (e.timestamp >= visited_time[u]) {
        int old_time = atomicMin(&visited_time[e.dest], e.timestamp);
        if (e.timestamp < old_time) {
            int pos = atomicAdd(next_frontier_size, 1);
            if (pos < num_nodes) {
                next_frontier[pos] = e.dest;
            }
        }
    }
}
}


void run_temporal_bfs(
    int start_node, int num_nodes,
    thrust::device_vector<int>& d_row_offsets,
    thrust::device_vector<Edge>& d_edges
) {
    thrust::device_vector<int> visited_time(num_nodes, INT_MAX);
    thrust::device_vector<int> frontier(num_nodes, -1);
    thrust::device_vector<int> next_frontier(num_nodes, -1);

    // BFS initialization
    thrust::fill(frontier.begin(), frontier.end(), -1);
    frontier[0] = start_node;
    visited_time[start_node] = 0;

    int* d_row = thrust::raw_pointer_cast(d_row_offsets.data());
    Edge* d_edge = thrust::raw_pointer_cast(d_edges.data());
    int* d_visited = thrust::raw_pointer_cast(visited_time.data());
    int* d_frontier = thrust::raw_pointer_cast(frontier.data());
    int* d_next = thrust::raw_pointer_cast(next_frontier.data());

    int current_time = 0;
    int frontier_size = 1;

    thrust::device_vector<int> d_frontier_size(1);

    while (frontier_size > 0) {
        // Reset next frontier size to 0
        thrust::fill(d_frontier_size.begin(), d_frontier_size.end(), 0);
        int* d_next_size = thrust::raw_pointer_cast(d_frontier_size.data());

        // Launch kernel
        int blockSize = 128;
        int numBlocks = (frontier_size + blockSize - 1) / blockSize;

        
        // thrust::fill(d_frontier_size.begin(), d_frontier_size.end(), 0);
        temporal_bfs_kernel<<<numBlocks, blockSize>>>(
            d_row, d_edge,
            d_frontier, d_next,
            d_visited, d_next_size,
            frontier_size, current_time,
            num_nodes
        );
        cudaDeviceSynchronize();


        cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);


        if (frontier_size == 0) break;
        if (frontier_size > num_nodes) { 
            std::cerr << "Error: frontier_size (" << frontier_size << ") exceeded num_nodes (" << num_nodes << "). Truncating." << std::endl;
            frontier_size = num_nodes;
        }



        thrust::copy(next_frontier.begin(), next_frontier.begin() + frontier_size, frontier.begin());
        
        current_time++;
    }

    // Copy back and print visited times
    std::vector<int> h_times(num_nodes);
    thrust::copy(visited_time.begin(), visited_time.end(), h_times.begin());

    std::cout << "Earliest arrival times:\n";
    for (int i = 0; i < num_nodes; ++i) {
        if (h_times[i] == INT_MAX)
            std::cout << "Node " << i << ": unreachable\n";
        else
            std::cout << "Node " << i << ": " << h_times[i] << "\n";
    }
}


int main() {
    // Input edge list: (u, v, t, w)
    thrust::device_vector<int4> input_edges_device; // Renamed to avoid conflict
    std::vector<int4> h_input_edges; // Host vector to read from file

    std::string filename = "data/small_temporal_graph.csv"; 
    std::ifstream file(filename);
    std::string line;
    int max_node_id = -1;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        // Fallback to hardcoded data if file not found
        std::cout << "Falling back to hardcoded graph data." << std::endl;
        h_input_edges = {
            int4{0, 1, 1, 2}, 
            int4{0, 1, 2, 3}, 
            int4{1, 3, 3, 1},
            int4{1, 4, 4, 5},
            int4{2, 3, 3, 2},
            int4{2, 5, 5, 1},
            int4{3, 6, 6, 2},
            int4{4, 6, 6, 1},
            int4{5, 6, 6, 2},
            int4{5, 7, 7, 3},
            int4{6, 7, 8, 1},
            int4{7, 0, 10, 4},  
            int4{1, 5, 5, 2},
            int4{0, 4, 3, 4},
            int4{3, 5, 4, 1},
            int4{6, 2, 9, 2}  
        };
        for (const auto& edge : h_input_edges) {
            if (edge.x > max_node_id) max_node_id = edge.x;
            if (edge.y > max_node_id) max_node_id = edge.y;
        }
    } else {
        std::cout << "Reading graph data from " << filename << std::endl;
        while (std::getline(file, line)) {
            if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
                // Skip empty or whitespace-only lines
                continue;
            }
            std::stringstream ss(line);
            std::string segment;
            int u, v, w, t;

            try {
                // Assuming CSV format: u,v,w,t
                if (std::getline(ss, segment, ',')) u = std::stoi(segment); else { std::cerr << "Error parsing u from line: " << line << std::endl; continue; }
                if (std::getline(ss, segment, ',')) v = std::stoi(segment); else { std::cerr << "Error parsing v from line: " << line << std::endl; continue; }
                if (std::getline(ss, segment, ',')) w = std::stoi(segment); else { std::cerr << "Error parsing w from line: " << line << std::endl; continue; }
                if (std::getline(ss, segment, ',')) t = std::stoi(segment); else { std::cerr << "Error parsing t from line: " << line << std::endl; continue; }
            } catch (const std::invalid_argument& ia) {
                std::cerr << "Invalid argument: " << ia.what() << " on line: " << line << std::endl;
                continue; // Skip this line and try the next
            } catch (const std::out_of_range& oor) {
                std::cerr << "Out of range: " << oor.what() << " on line: " << line << std::endl;
                continue; // Skip this line
            }

            h_input_edges.push_back(int4{u, v, t, w});
            if (u > max_node_id) max_node_id = u;
            if (v > max_node_id) max_node_id = v;
        }
        file.close();
    }

    if (h_input_edges.empty()) {
        std::cerr << "Error: No edges loaded. Exiting." << std::endl;
        return 1;
    }
    
    // Copy host vector to device vector
    input_edges_device = h_input_edges;


    int num_nodes = (max_node_id == -1) ? 8 : max_node_id + 1; // Determine num_nodes dynamically
    std::cout << "Number of nodes detected/set: " << num_nodes << std::endl;
    std::cout << "Number of edges loaded: " << input_edges_device.size() << std::endl;

    thrust::device_vector<int> d_row_offsets;
    thrust::device_vector<Edge> d_edges;

    // Convert to CSR
    device_convert_to_csr(input_edges_device, num_nodes, d_row_offsets, d_edges);

    // Output for verification
    // thrust::host_vector<int> h_offsets = d_row_offsets;
    // thrust::host_vector<Edge> h_edges = d_edges;

    std::vector<int> h_offsets(d_row_offsets.size());
    std::vector<Edge> h_edges(d_edges.size());
    
    thrust::copy(d_row_offsets.begin(), d_row_offsets.end(), h_offsets.begin());
    thrust::copy(d_edges.begin(), d_edges.end(), h_edges.begin());


    std::cout << "row_offsets:\n";
    for (int i = 0; i < h_offsets.size(); ++i) {
        std::cout << h_offsets[i] << " ";
    }
    std::cout << "\nedges (dest, timestamp, weight):\n";
    for (const auto& e : h_edges) {
        std::cout << e.dest << " " << e.timestamp << " " << e.weight << "\n";
    }

    run_temporal_bfs(0, num_nodes, d_row_offsets, d_edges);

    return 0;
}

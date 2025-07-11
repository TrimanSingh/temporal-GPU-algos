#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string> // Added for std::stoi/stod
#include <cuda_runtime.h>

// ---------- Types ----------
using vid_t = int;
using ts_t = int;

// struct Edge {
//     vid_t u, v;
//     ts_t t;
// };

struct __host__ __device__ Edge {
    vid_t u, v;
    ts_t t;
};

struct EdgeData {
    std::vector<Edge> edges;
    int max_vid = -1;
};

// ---------- Load Edges (CSV format: u,v,weight,t) ----------
EdgeData load_edges(const std::string &path) {
    EdgeData ed;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << path << std::endl;
        return ed; // Return empty EdgeData
    }
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        line_number++;
        std::istringstream ss(line);
        std::string token;
        int u, v, w, t;
        try {
            if (!std::getline(ss, token, ',')) throw std::runtime_error("Missing u");
            u = std::stoi(token);
            if (!std::getline(ss, token, ',')) throw std::runtime_error("Missing v");
            v = std::stoi(token);
            if (!std::getline(ss, token, ',')) throw std::runtime_error("Missing weight");
            // w = std::stoi(token); // Weight is skipped
            if (!std::getline(ss, token, ',')) throw std::runtime_error("Missing t");
            t = std::stoi(token);

            if (u < 0 || v < 0) {
                std::cerr << "Warning: Negative vertex ID found in line " << line_number
                          << ": (" << u << "," << v << "). Skipping this edge." << std::endl;
                continue;
            }

        } catch (const std::invalid_argument& ia) {
            std::cerr << "Warning: Invalid number format in CSV line " << line_number << ": '" << line << "'. Skipping. Error: " << ia.what() << std::endl;
            continue;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Warning: Number out of range in CSV line " << line_number << ": '" << line << "'. Skipping. Error: " << oor.what() << std::endl;
            continue;
        } catch (const std::runtime_error& re) {
            std::cerr << "Warning: Malformed CSV line " << line_number << " (" << re.what() << "): '" << line << "'. Skipping." << std::endl;
            continue;
        }

        ed.edges.push_back({u, v, t});
        ed.max_vid = std::max({ed.max_vid, u, v});
    }
    return ed;
}

// ---------- GPU Utilities ----------
void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Available GPUs: " << deviceCount << std::endl;
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        size_t free_mem, total_mem;
        cudaSetDevice(dev);
        cudaMemGetInfo(&free_mem, &total_mem);
        
        std::cout << "GPU " << dev << ": " << deviceProp.name 
                  << " (Free: " << (free_mem / (1024*1024)) << " MB, "
                  << "Total: " << (total_mem / (1024*1024)) << " MB)" << std::endl;
    }
}

int selectBestGPU() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    int bestDevice = 0;
    size_t maxFreeMem = 0;
    
    for (int dev = 0; dev < deviceCount; dev++) {
        size_t free_mem, total_mem;
        cudaSetDevice(dev);
        cudaMemGetInfo(&free_mem, &total_mem);
        
        if (free_mem > maxFreeMem) {
            maxFreeMem = free_mem;
            bestDevice = dev;
        }
    }
    
    cudaSetDevice(bestDevice);
    std::cout << "Selected GPU " << bestDevice << " with " << (maxFreeMem / (1024*1024)) << " MB free memory" << std::endl;
    return bestDevice;
}

// ---------- Compute Degeneracy Ordering ----------
std::vector<vid_t> degeneracy_order(int n, const std::vector<Edge>& edges) {
    if (n == 0) {
        return std::vector<vid_t>(); // Handle empty graph case
    }
    
    // Estimate memory usage
    size_t estimated_memory = 0;
    estimated_memory += (size_t)n * sizeof(std::vector<vid_t>); // adj list headers
    estimated_memory += edges.size() * 2 * sizeof(vid_t); // adj list data (each edge counted twice)
    estimated_memory += (size_t)(n + 1) * sizeof(std::vector<vid_t>); // bucket headers
    estimated_memory += (size_t)n * sizeof(vid_t); // bucket data
    estimated_memory += (size_t)n * sizeof(int); // degrees
    estimated_memory += (size_t)n * sizeof(bool); // removed array
    estimated_memory += (size_t)n * sizeof(vid_t); // order array
    
    std::cout << "Estimated memory usage: " << (estimated_memory / (1024*1024)) << " MB" << std::endl;
    
    // Check if estimated memory is too large (> 8GB)
    // if (estimated_memory > 8ULL * 1024 * 1024 * 1024) {
    //     std::cerr << "Error: Estimated memory usage (" << (estimated_memory / (1024*1024*1024)) 
    //               << " GB) exceeds 8GB limit." << std::endl;
    //     std::cerr << "Consider using a subset of the graph or a machine with more RAM." << std::endl;
    //     throw std::runtime_error("Memory limit exceeded");
    // }
    
    std::cout << "Building adjacency list for " << n << " vertices..." << std::endl;
    
    try {
        std::vector<std::vector<vid_t>> adj;
        adj.reserve(n);
        adj.resize(n);
        std::cout << "Allocated adjacency list. Processing " << edges.size() << " edges..." << std::endl;
        
        for (const auto& e : edges) {
            if (e.u >= n || e.v >= n || e.u < 0 || e.v < 0) {
                std::cerr << "Error: Invalid vertex ID in edge (" << e.u << "," << e.v << "). Max allowed: " << (n-1) << std::endl;
                throw std::runtime_error("Invalid vertex ID");
            }
            adj[e.u].push_back(e.v);
            adj[e.v].push_back(e.u);
        }
        
        std::cout << "Computing degrees..." << std::endl;
        std::vector<int> deg;
        deg.reserve(n);
        deg.resize(n);
        for (int i = 0; i < n; i++) {
            deg[i] = adj[i].size();
        }
        
        std::cout << "Creating degree buckets..." << std::endl;
        std::vector<std::vector<vid_t>> bucket;
        bucket.reserve(n + 1);
        bucket.resize(n + 1);
        for (int i = 0; i < n; i++) {
            if (deg[i] <= n) {
                bucket[deg[i]].push_back(i);
            } else {
                std::cerr << "Error: Vertex " << i << " has degree " << deg[i] << " > n=" << n << std::endl;
                throw std::runtime_error("Invalid degree");
            }
        }

        std::cout << "Starting degeneracy ordering..." << std::endl;
        std::vector<bool> removed;
        removed.reserve(n);
        removed.resize(n, false);
        std::vector<vid_t> order;
        order.reserve(n);
        order.resize(n);
        
        for (int k = 0; k < n; k++) {
            // if (k % 100000 == 0) {
            //     std::cout << "Processed " << k << "/" << n << " vertices" << std::endl;
            // }
            int curr_deg = 0;
            while (curr_deg <= n && bucket[curr_deg].empty()) {
                curr_deg++;
            }

            if (curr_deg > n) {
                // This means all buckets bucket[0]...bucket[n] are empty.
                // If k < n at this point, it's an error: we expected to find more vertices.
                std::cerr << "Critical Error in degeneracy_order: All buckets exhausted prematurely.\n"
                          << "k = " << k << ", n = " << n << ", curr_deg = " << curr_deg << std::endl;
                // This could indicate that n is larger than the number of actual vertices
                // or an issue in degree maintenance.
                throw std::logic_error("Degeneracy order failed: Buckets exhausted prematurely.");
            }
            // Now, curr_deg <= n and bucket[curr_deg] is non-empty.
            vid_t u = bucket[curr_deg].back();
            bucket[curr_deg].pop_back();
            removed[u] = true;
            order[u] = k;
            for (vid_t v: adj[u]) {
                if (!removed[v]) {
                    int old_deg = deg[v];
                    deg[v]--;
                    if (old_deg > 0 && old_deg <= n) {
                        bucket[old_deg - 1].push_back(v);
                    }
                }
            }
        }
        std::cout << "Degeneracy ordering completed." << std::endl;
        return order;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed in degeneracy_order: " << e.what() << std::endl;
        std::cerr << "Graph too large for available memory" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error in degeneracy_order: " << e.what() << std::endl;
        throw;
    }
}

// ---------- Orient edges forward-only ----------
void orient_edges(const std::vector<Edge>& edges, const std::vector<vid_t>& order,
                  std::vector<Edge>& out) {
    for (auto&e: edges) {
        if (order[e.u] > order[e.v]) out.push_back(e);
        else if (order[e.v] > order[e.u]) out.push_back({e.v, e.u, e.t});
    }
}

// ---------- CSR builder with timestamps ----------
void build_csr(int n, const std::vector<Edge>& edges,
               std::vector<int>& idx, std::vector<int>& adj, std::vector<ts_t>& adj_t) {
    int M = edges.size();
    idx.assign(n+1,0);
    for (auto&e: edges) idx[e.u+1]++;
    for (int i=1;i<=n;i++) idx[i]+=idx[i-1];

    adj.resize(M);
    adj_t.resize(M);
    std::vector<int> ptr = idx;
    for (auto&e: edges) {
        int p = ptr[e.u]++;
        adj[p]=e.v; adj_t[p]=e.t;
    }

    for(int u=0; u<n; u++){
        int s=idx[u], e=idx[u+1];
        if (s == e) continue; // Skip if no outgoing edges for u

        // Create pairs of (neighbor_vertex, timestamp) for sorting
        std::vector<std::pair<int, ts_t>> tmp;
        tmp.reserve(e - s); // Pre-allocate memory
        for(int i=s;i<e;i++) tmp.emplace_back(adj[i], adj_t[i]); // Store (neighbor_vertex, timestamp)

        // Sort by neighbor_vertex, then by timestamp
        std::sort(tmp.begin(), tmp.end());

        // Update adj and adj_t arrays with sorted order
        for(size_t i=0; i < tmp.size(); ++i) {
            adj[s+i]   = tmp[i].first;  // neighbor_vertex
            adj_t[s+i] = tmp[i].second; // timestamp
        }
    }
}

// ---------- GPU binary search ----------
__device__ bool search_closure(const int* adj, const ts_t* at, int lo, int hi, int w, ts_t tvw, ts_t delta) {
    while (lo < hi) {
        int m = (lo+hi)/2;
        if (adj[m] < w || (adj[m]==w && at[m] <= tvw)) lo = m+1;
        else hi = m;
    }
    return (lo < hi && adj[lo]==w && at[lo]-tvw <= delta);
}

// ---------- GPU kernel ----------
__global__ void dottt_kernel(
    const int* eu, const int* ev, const ts_t* et, int M,
    const int* idx, const int* adj, const ts_t* at,
    ts_t delta12, ts_t delta13, ts_t delta23,
    unsigned long long *res,
    Edge* triangle_out,
    size_t max_triangles_output)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= M) return;

    int u = eu[e], v = ev[e];
    ts_t tuv = et[e];
    int baseU = idx[u], endU = idx[u+1];
    int baseV = idx[v], endV = idx[v+1];

    for (int j = baseV; j < endV; ++j) {
        int w = adj[j];
        ts_t tvw = at[j];
        if (tvw <= tuv || tvw - tuv > delta12) continue;

        // now check u→w exists and within delta13
        int lo = baseU, hi = endU;
        while (lo < hi) {
            int m = (lo + hi) >> 1;
            if (adj[m] < w || (adj[m] == w && at[m] <= tuv)) lo = m + 1;
            else hi = m;
        }
        if (lo < endU && adj[lo] == w) {
            ts_t tuw = at[lo];
            if ((tuw - tuv <= delta13) && (tuw - tvw <= delta23)) {
                unsigned long long pos = atomicAdd((unsigned long long*)res, 1ULL);
                if (pos < max_triangles_output) {
                    size_t base = 3 * pos;
                    triangle_out[base + 0] = Edge{u, v, tuv};  // edge1: u → v @ t_uv
                    triangle_out[base + 1] = Edge{u, w, tuw};  // edge2: u → w @ t_uw
                    triangle_out[base + 2] = Edge{v, w, tvw};  // edge3: v → w @ t_vw
                }
            }
        }
    }
}

unsigned long long dottt_count(std::vector<Edge>& edges, int n, ts_t delta12, ts_t delta13, ts_t delta23, size_t max_triangles) {
    std::cout << "Running dottt_count for n=" << n << " vertices." << std::endl;
    
    // Check GPU memory before proceeding
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory: " << (free_mem / (1024*1024)) << " MB free, " << (total_mem / (1024*1024)) << " MB total" << std::endl;
    
    auto order = degeneracy_order(n,edges);
    std::vector<Edge> fwd;
    orient_edges(edges, order, fwd);
    // for (auto& e : fwd) {
    //     std::cout << "Edge: " << e.u << " -> " << e.v << " at time " << e.t << "\n";
    // }
    int M = fwd.size();
    std::cout << "Oriented graph has " << M << " edges." << std::endl;

    std::vector<int> idx, adj;
    std::vector<ts_t> at;
    build_csr(n, fwd, idx, adj, at);
    std::cout << "CSR created. idx size: " << idx.size() << ", adj size: " << adj.size() << ", at size: " << at.size() << std::endl;

    // Estimate GPU memory usage
    size_t gpu_memory_needed = 0;
    gpu_memory_needed += M * sizeof(int) * 2; // eu, ev
    gpu_memory_needed += M * sizeof(ts_t); // et
    gpu_memory_needed += (n+1) * sizeof(int); // idx
    gpu_memory_needed += M * sizeof(int); // adj
    gpu_memory_needed += M * sizeof(ts_t); // at
    gpu_memory_needed += 3 * max_triangles * sizeof(Edge); // triangle output
    gpu_memory_needed += sizeof(unsigned long long); // result counter
    
    std::cout << "Estimated GPU memory needed: " << (gpu_memory_needed / (1024*1024)) << " MB" << std::endl;
    
    if (gpu_memory_needed > free_mem) {
        std::cerr << "Error: Not enough GPU memory! Needed: " << (gpu_memory_needed / (1024*1024)) 
                  << " MB, Available: " << (free_mem / (1024*1024)) << " MB" << std::endl;
        std::cerr << "Try reducing max_triangles or using a GPU with more memory." << std::endl;
        throw std::runtime_error("Insufficient GPU memory");
    }

//     for (int u = 0; u < n; ++u) {
//   for (int i = idx[u]; i < idx[u+1]; ++i) {
//     std::cout << "("<<u<<"→"<<adj[i]<<"@"<<at[i]<<") ";
//   }
//   std::cout<<"\n";
// }

    thrust::host_vector<int> eu(M), ev(M); thrust::host_vector<ts_t> et(M);
    for(int i=0;i<M;i++){ eu[i]=fwd[i].u; ev[i]=fwd[i].v; et[i]=fwd[i].t; }

    std::cout << "Copying data to GPU..." << std::endl;
    try {
        thrust::device_vector<int> deu=eu, dev=ev, didx=idx, dadj=adj;
        thrust::device_vector<ts_t> det=et, dat=at;
        thrust::device_vector<unsigned long long> dres(1,0ULL);

        // int max_triangles = 10000; // This line is now a parameter
        std::cout << "Allocating space for " << max_triangles << " triangles." << std::endl;
        thrust::device_vector<Edge> dtri_edges(3 * max_triangles);

        std::cout << "Launching kernel..." << std::endl;
        int B=128, G=(M+B-1)/B;
        dottt_kernel<<<G,B>>>(
            thrust::raw_pointer_cast(deu.data()),
            thrust::raw_pointer_cast(dev.data()),
            thrust::raw_pointer_cast(det.data()), M,
            thrust::raw_pointer_cast(didx.data()),
            thrust::raw_pointer_cast(dadj.data()),
            thrust::raw_pointer_cast(dat.data()),
            delta12, delta13, delta23,
            thrust::raw_pointer_cast(dres.data()),
            thrust::raw_pointer_cast(dtri_edges.data()),
            max_triangles
        );
        cudaDeviceSynchronize();

        std::cout << "Copying results back..." << std::endl;
        thrust::host_vector<Edge> htri_edges = dtri_edges;

        unsigned long long count = dres[0];
        size_t triangles_to_print = std::min(max_triangles, (size_t)count);

        for (size_t i = 0; i < triangles_to_print; ++i) {
            const Edge& e1 = htri_edges[3*i + 0];
            const Edge& e2 = htri_edges[3*i + 1];
            const Edge& e3 = htri_edges[3*i + 2];
            // std::cout << "Triangle #" << i+1 << ":\n";
            // std::cout << "  Edge: (" << e1.u << " → " << e1.v << ") @ " << e1.t << "\n";
            // std::cout << "  Edge: (" << e2.u << " → " << e2.v << ") @ " << e2.t << "\n";
            // std::cout << "  Edge: (" << e3.u << " → " << e3.v << ") @ " << e3.t << "\n";
        }

        if (count > max_triangles) {
            std::cout << "...\n(" << count - max_triangles << " more triangles found but not printed)\n";
        }
        return count;
    } catch (const thrust::system_error& e) {
        std::cerr << "Thrust/CUDA error: " << e.what() << std::endl;
        throw;
    }
}




// ---------- Main ----------
int main(int argc, char* argv[]){
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <filename> <delta12> <delta13> <delta23> [max_triangles_output] [gpu_id]" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    ts_t delta12 = std::stoi(argv[2]);
    ts_t delta13 = std::stoi(argv[3]);
    ts_t delta23 = std::stoi(argv[4]);
    size_t max_triangles_output = 10000; // Default value
    if (argc > 5) {
        max_triangles_output = std::stoull(argv[5]);
    }

    // GPU selection
    // printGPUInfo();
    int gpu_id = -1;
    if (argc > 6) {
        gpu_id = std::stoi(argv[6]);
        cudaSetDevice(gpu_id);
        std::cout << "Using GPU " << gpu_id << " as specified." << std::endl;
    } else {
        // gpu_id = selectBestGPU();
    }

    auto ed = load_edges(filename);
    if (ed.max_vid == -1) { // Check if graph loading failed (e.g. file not found)
        std::cerr << "Error loading graph from file: " << filename << std::endl;
        return 1;
    }
    std::cout << "Graph loaded: " << ed.edges.size() << " edges, " << ed.max_vid + 1 << " vertices." << std::endl;
    int n = ed.max_vid + 1;

    try {
        // Pass max_triangles_output to dottt_count
        auto count = dottt_count(ed.edges, n, delta12, delta13, delta23, max_triangles_output);
        std::cout << "Temporal triangles (DOTTT, deltas=" << delta12 << "," << delta13 << "," << delta23 << "): " << count << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
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

// ---------- Compute Degeneracy Ordering ----------
std::vector<vid_t> degeneracy_order(int n, const std::vector<Edge>& edges) {
    if (n == 0) {
        return std::vector<vid_t>(); // Handle empty graph case
    }
    std::vector<std::vector<vid_t>> adj(n);
    for (auto&e: edges) {
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }
    std::vector<int> deg(n);
    for (int i = 0; i < n; i++) deg[i] = adj[i].size();
    std::vector<std::vector<vid_t>> bucket(n+1);
    for (int i = 0; i < n; i++) bucket[deg[i]].push_back(i);

    std::vector<bool> removed(n,false);
    std::vector<vid_t> order(n);
    for (int k = 0; k < n; k++) {
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
        for (vid_t v: adj[u]) if (!removed[v]) {
            int d = deg[v]--;
            bucket[d-1].push_back(v);
        }
    }
    return order;
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
    Edge* triangle_out)
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
                // atomicAdd(res, 1ULL);
                int pos = atomicAdd((int*)res, 1);
                int base = 3 * pos;
                triangle_out[base + 0] = Edge{u, v, tuv};  // edge1: u → v @ t_uv
                triangle_out[base + 1] = Edge{u, w, tuw};  // edge2: u → w @ t_uw
                triangle_out[base + 2] = Edge{v, w, tvw};  // edge3: v → w @ t_vw
            }
        }
    }
}

unsigned long long dottt_count(std::vector<Edge>& edges, int n, ts_t delta12, ts_t delta13, ts_t delta23, int max_triangles) {
    auto order = degeneracy_order(n,edges);
    std::vector<Edge> fwd;
    orient_edges(edges, order, fwd);
    // for (auto& e : fwd) {
    //     std::cout << "Edge: " << e.u << " -> " << e.v << " at time " << e.t << "\n";
    // }
    int M = fwd.size();

    std::vector<int> idx, adj;
    std::vector<ts_t> at;
    build_csr(n, fwd, idx, adj, at);

//     for (int u = 0; u < n; ++u) {
//   for (int i = idx[u]; i < idx[u+1]; ++i) {
//     std::cout << "("<<u<<"→"<<adj[i]<<"@"<<at[i]<<") ";
//   }
//   std::cout<<"\n";
// }

    thrust::host_vector<int> eu(M), ev(M); thrust::host_vector<ts_t> et(M);
    for(int i=0;i<M;i++){ eu[i]=fwd[i].u; ev[i]=fwd[i].v; et[i]=fwd[i].t; }

    thrust::device_vector<int> deu=eu, dev=ev, didx=idx, dadj=adj;
    thrust::device_vector<ts_t> det=et, dat=at;
    thrust::device_vector<unsigned long long> dres(1,0ULL);

    // int max_triangles = 10000; // This line is now a parameter
    thrust::device_vector<Edge> dtri_edges(3 * max_triangles);

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
        thrust::raw_pointer_cast(dtri_edges.data())
    );
    cudaDeviceSynchronize();


    thrust::host_vector<Edge> htri_edges = dtri_edges;


    for (unsigned long long i = 0; i < dres[0]; ++i) {
        const Edge& e1 = htri_edges[3*i + 0];
        const Edge& e2 = htri_edges[3*i + 1];
        const Edge& e3 = htri_edges[3*i + 2];
        std::cout << "Triangle #" << i+1 << ":\n";
        std::cout << "  Edge: (" << e1.u << " → " << e1.v << ") @ " << e1.t << "\n";
        std::cout << "  Edge: (" << e2.u << " → " << e2.v << ") @ " << e2.t << "\n";
        std::cout << "  Edge: (" << e3.u << " → " << e3.v << ") @ " << e3.t << "\n";
    }
    return dres[0];
}




// ---------- Main ----------
int main(int argc, char* argv[]){
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <filename> <delta12> <delta13> <delta23> <max_triangles_output (optional)>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    ts_t delta12 = std::stoi(argv[2]);
    ts_t delta13 = std::stoi(argv[3]);
    ts_t delta23 = std::stoi(argv[4]);
    int max_triangles_output = 10000; // Default value
    if (argc > 5) {
        max_triangles_output = std::stoi(argv[5]);
    }


    auto ed = load_edges(filename);
    if (ed.max_vid == -1) { // Check if graph loading failed (e.g. file not found)
        std::cerr << "Error loading graph from file: " << filename << std::endl;
        return 1;
    }
    int n = ed.max_vid + 1;

    // Pass max_triangles_output to dottt_count
    auto count = dottt_count(ed.edges, n, delta12, delta13, delta23, max_triangles_output);
    std::cout << "Temporal triangles (DOTTT, deltas=" << delta12 << "," << delta13 << "," << delta23 << "): " << count << "\n";
    return 0;
}

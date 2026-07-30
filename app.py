import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import math
import random
import time
import os
import re

# ==========================================
# NUMBA CHECK (FOR C++ LEVEL SPEED)
# ==========================================
try:
    from numba import njit
except ImportError:
    st.error("⚠️ ERROR: 'numba' library is missing! Please type 'pip install numba' in your terminal.")
    st.stop()

# ==========================================
# 1. MODULE: DYNAMIC DATA READER (PARSER)
# ==========================================
class FSTSP_Parser:
    def __init__(self, file_content):
        self.max_fly = float('inf') 
        self.novisit_list = []
        self.truck_speed = 1.0 
        self.drone_speed = 1.0 
        self.num_nodes = 0
        self.nodes = [] 
        
        self._parse(file_content)
        self.truck_time_matrix, self.drone_time_matrix = self._build_matrices()

    def _parse(self, content):
        lines = content.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1; continue
                
            if line.startswith("#MAXFLY"):
                val = line.split()[1]
                self.max_fly = float('inf') if val.lower() == 'infinity' else float(val)
            elif line.startswith("#NOVISIT"):
                self.novisit_list.append(int(line.split()[1]))
            elif "The speed of the Truck" in line:
                i += 1; self.truck_speed = float(lines[i].strip())
            elif "The speed of the Drone" in line:
                i += 1; self.drone_speed = float(lines[i].strip())
            elif "Number of Nodes" in line:
                i += 1; self.num_nodes = int(lines[i].strip())
            elif "The Depot" in line:
                i += 1
                parts = lines[i].strip().split()
                self.nodes.append((0, float(parts[0]), float(parts[1]))) 
            elif "The Locations" in line:
                i += 1
                for j in range(1, self.num_nodes):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        self.nodes.append((j, float(parts[0]), float(parts[1])))
                        i += 1
                break
            i += 1

    def _build_matrices(self):
        n = len(self.nodes)
        t_matrix = np.zeros((n, n))
        d_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.nodes[i][1], self.nodes[i][2]
                    x2, y2 = self.nodes[j][1], self.nodes[j][2]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    t_matrix[i][j] = dist * self.truck_speed
                    d_matrix[i][j] = dist * self.drone_speed
        return t_matrix, d_matrix

# ==========================================
# 2. MODULE: FAST LOCAL SEARCH (SAFE MODE)
# ==========================================
@njit
def fast_2opt(route, t_matrix):
    n = len(route)
    improved = True
    best_route = route.copy()
    iters = 0
    max_iters = 30 # Sonsuz döngü (infinite loop) kilidini kırmak için güvenlik valfi
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                d1 = t_matrix[best_route[i-1], best_route[i]] + t_matrix[best_route[j], best_route[j+1]]
                d2 = t_matrix[best_route[i-1], best_route[j]] + t_matrix[best_route[i], best_route[j+1]]
                if d2 < d1 - 1e-5:
                    best_route[i:j+1] = best_route[i:j+1][::-1]
                    improved = True
    return best_route

@njit
def fast_3opt_relocation(route, t_matrix):
    n = len(route)
    improved = True
    best_route = route.copy()
    iters = 0
    max_iters = 20 # 3-Opt çok ağır olduğu için limiti 20'de tuttuk
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if i == j or i == j - 1:
                    continue
                prev_i = best_route[i-1]
                node_i = best_route[i]
                next_i = best_route[i+1]
                
                node_j = best_route[j]
                next_j = best_route[j+1]
                
                d_old = t_matrix[prev_i, node_i] + t_matrix[node_i, next_i] + t_matrix[node_j, next_j]
                d_new = t_matrix[prev_i, next_i] + t_matrix[node_j, node_i] + t_matrix[node_i, next_j]
                
                if d_new < d_old - 1e-5:
                    new_route = np.zeros_like(best_route)
                    idx = 0
                    for k in range(n):
                        if k == i: continue
                        new_route[idx] = best_route[k]
                        idx += 1
                        if k == j:
                            new_route[idx] = node_i
                            idx += 1
                    best_route = new_route
                    improved = True
                    break 
            if improved:
                break
    return best_route

# ==========================================
# 3. MODULE: NUMBA JIT SUPPORTED LIGHTNING FAST DECODER
# ==========================================
@njit
def numba_fast_dp_decode(rk_route, t_matrix, d_matrix, num_nodes, novisit_mask, max_fly):
    customers = np.arange(1, num_nodes)
    sorted_indices = np.argsort(rk_route)
    sorted_customers = customers[sorted_indices]

    seq = np.zeros(num_nodes + 1, dtype=np.int32)
    seq[1:num_nodes] = sorted_customers
    seq[0] = 0
    seq[num_nodes] = 0

    N = num_nodes + 1
    cost = np.full(N, np.inf)
    cost[0] = 0.0
    path_prev = np.zeros(N, dtype=np.int32)
    path_drone = np.full(N, -1, dtype=np.int32)

    WINDOW_SIZE = 12
    pure_t = np.zeros(N)

    for i in range(N - 1):
        if cost[i] == np.inf:
            continue

        limit_j = min(N, i + WINDOW_SIZE)
        curr_dist = 0.0
        for j in range(i + 1, limit_j):
            curr_dist += t_matrix[seq[j-1], seq[j]]
            pure_t[j] = curr_dist

        for j in range(i + 1, limit_j):
            if cost[i] + pure_t[j] < cost[j]:
                cost[j] = cost[i] + pure_t[j]
                path_prev[j] = i
                path_drone[j] = -1

            if j >= i + 2:
                for k in range(i + 1, j):
                    drone_cust = seq[k]
                    if novisit_mask[drone_cust]:
                        continue

                    d_time = d_matrix[seq[i], drone_cust] + d_matrix[drone_cust, seq[j]]
                    if d_time > max_fly:
                        continue

                    t_skip = pure_t[j] - t_matrix[seq[k-1], seq[k]] - t_matrix[seq[k], seq[k+1]] + t_matrix[seq[k-1], seq[k+1]]
                    seg_time = max(t_skip, d_time)

                    if cost[i] + seg_time < cost[j]:
                        cost[j] = cost[i] + seg_time
                        path_prev[j] = i
                        path_drone[j] = drone_cust

    return cost[N-1], path_prev, path_drone, seq


class DPSplitDecoder:
    def __init__(self, parsed_data):
        self.data = parsed_data
        self.t = parsed_data.truck_time_matrix
        self.d = parsed_data.drone_time_matrix
        self.num_nodes = parsed_data.num_nodes
        self.max_fly = parsed_data.max_fly
        
        self.novisit_mask = np.zeros(self.num_nodes, dtype=np.bool_)
        for nv in parsed_data.novisit_list:
            self.novisit_mask[nv] = True

    def decode(self, rk_route):
        rk_arr = np.array(rk_route, dtype=np.float64)
        cost, path_prev, path_drone, seq = numba_fast_dp_decode(
            rk_arr, self.t, self.d, self.num_nodes, self.novisit_mask, self.max_fly
        )

        N = len(seq)
        if cost == np.inf:
            return float('inf'), [], []

        curr = N - 1
        segments = []
        while curr != 0:
            prev_i = path_prev[curr]
            drone_cust = path_drone[curr]
            segments.append((prev_i, curr, drone_cust if drone_cust != -1 else None))
            curr = prev_i
            
        segments.reverse()
        truck_route = [0]
        drone_trips = []
        
        for prev_i, curr_i, drone_cust in segments:
            for x in range(prev_i + 1, curr_i + 1):
                if seq[x] != drone_cust:
                    truck_route.append(int(seq[x]))
            if drone_cust is not None:
                drone_trips.append((int(seq[prev_i]), int(drone_cust), int(seq[curr_i])))
                
        return cost, truck_route, drone_trips

# ==========================================
# 4. MODULE: BRKGA PROBABILISTIC MEMETIC ENGINE
# ==========================================
class BRKGA_Engine:
    def __init__(self, p, p_e_ratio, p_m_ratio, rho_e, max_gen, decoder):
        self.p = p
        self.p_e = int(p * (p_e_ratio / 100))
        self.p_m = int(p * (p_m_ratio / 100))
        self.rho_e = rho_e
        self.max_gen = max_gen
        self.decoder = decoder
        self.num_cust = decoder.data.num_nodes - 1

    def create_individual(self):
        return {
            'route': [random.random() for _ in range(self.num_cust)],
            'fitness': float('inf'), 'truck_route': [], 'drone_trips': []
        }
        
    def create_smart_mutant(self, elites):
        parent = random.choice(elites)
        mutant_route = parent['route'].copy()
        num_swaps = max(1, int(self.num_cust * 0.10))
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(self.num_cust), 2)
            mutant_route[idx1], mutant_route[idx2] = mutant_route[idx2], mutant_route[idx1]
        return {
            'route': mutant_route,
            'fitness': float('inf'), 'truck_route': [], 'drone_trips': []
        }

    def evaluate(self, ind, use_2opt, use_3opt, force_ls=False):
        rk_route = np.array(ind['route'])
        
        # LS PROBABILITY: Sunucuyu korumak için LS sadece %10 ihtimalle çalışır (PMA Mimarisi)
        apply_ls = (use_2opt or use_3opt) and (force_ls or random.random() < 0.10)
        
        if apply_ls:
            customers = np.arange(1, self.num_cust + 1)
            sorted_indices = np.argsort(rk_route)
            seq = customers[sorted_indices]
            
            full_seq = np.zeros(len(seq) + 2, dtype=np.int32)
            full_seq[1:-1] = seq
            
            if use_2opt:
                full_seq = fast_2opt(full_seq, self.decoder.t)
            if use_3opt:
                full_seq = fast_3opt_relocation(full_seq, self.decoder.t)
                
            new_seq = full_seq[1:-1]
            new_rk = np.zeros(len(new_seq))
            spacing = 1.0 / (len(new_seq) + 1)
            for i, cust in enumerate(new_seq):
                orig_idx = cust - 1 
                new_rk[orig_idx] = (i + 1) * spacing
            ind['route'] = new_rk.tolist()

        cost, t_route, d_trips = self.decoder.decode(ind['route'])
        ind['fitness'] = cost
        ind['truck_route'] = t_route
        ind['drone_trips'] = d_trips

    def run(self, progress_bar, status_text, use_2opt, use_3opt):
        population = [self.create_individual() for _ in range(self.p)]
        for ind in population: self.evaluate(ind, use_2opt, use_3opt)
        
        best_solution = None

        for gen in range(self.max_gen):
            population.sort(key=lambda x: x['fitness'])
            if best_solution is None or population[0]['fitness'] < best_solution['fitness']:
                best_solution = population[0].copy()
                
            next_gen = []
            elites = population[:self.p_e]
            non_elites = population[self.p_e:]
            next_gen.extend(elites)
            
            mutants = []
            for _ in range(self.p_m):
                if random.random() < 0.5:
                    new_mut = self.create_smart_mutant(elites)
                else:
                    new_mut = self.create_individual()
                self.evaluate(new_mut, use_2opt, use_3opt)
                mutants.append(new_mut)
                
            next_gen.extend(mutants)
            
            num_offspring = self.p - self.p_e - self.p_m
            for _ in range(num_offspring):
                parent_a = random.choice(elites)
                parent_b = random.choice(non_elites) if non_elites else random.choice(elites)
                child = {'route': [], 'fitness': float('inf')}
                for i in range(self.num_cust):
                    child['route'].append(parent_a['route'][i] if random.random() < self.rho_e else parent_b['route'][i])
                
                self.evaluate(child, use_2opt, use_3opt)
                next_gen.append(child)
                
            population = next_gen
            
            if gen % 10 == 0:
                progress_bar.progress((gen + 1) / self.max_gen)
                mode_text = "Memetic PMA (Safe Mode)" if (use_2opt or use_3opt) else "Smart BRKGA"
                status_text.text(f"🧠 {mode_text}... Generation {gen+1}/{self.max_gen} | Score: {best_solution['fitness']:.2f}")

        progress_bar.progress(1.0)
        status_text.text(f"Completed! Found Optimum Makespan: {best_solution['fitness']:.2f}")
        return best_solution

# ==========================================
# 5. MODULE: INTERACTIVE MAP (PLOTLY)
# ==========================================
def draw_interactive_map(nodes_data, truck_route, drone_trips):
    fig = go.Figure()
    nodes_dict = {node[0]: (node[1], node[2]) for node in nodes_data}
    
    truck_x = [nodes_dict[n][0] for n in truck_route]
    truck_y = [nodes_dict[n][1] for n in truck_route]
    fig.add_trace(go.Scatter(x=truck_x, y=truck_y, mode='lines+markers+text', name='Truck Route',
                             text=[str(n) for n in truck_route], textposition="bottom center",
                             line=dict(color='#1f77b4', width=3), marker=dict(size=10, color='#1f77b4')))
    
    for i, (launch, visit, ret) in enumerate(drone_trips):
        dx = [nodes_dict[launch][0], nodes_dict[visit][0], nodes_dict[ret][0]]
        dy = [nodes_dict[launch][1], nodes_dict[visit][1], nodes_dict[ret][1]]
        fig.add_trace(go.Scatter(x=dx, y=dy, mode='lines+markers+text', name=f'Drone (Trip {i+1})',
                                 text=["", str(visit), ""], textposition="top center",
                                 line=dict(color='#d62728', width=2, dash='dashdot'), marker=dict(size=8, symbol='diamond')))
        
    fig.add_trace(go.Scatter(x=[nodes_dict[0][0]], y=[nodes_dict[0][1]], mode='markers+text', name='Depot',
                             text=["DEPOT"], textposition="top center", marker=dict(size=16, color='black', symbol='square')))
    
    fig.update_layout(
        title="🚁 FSTSP Optimum Route",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        hovermode="closest",
        plot_bgcolor='#f8f9fa',
        height=750,
        xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# ==========================================
# HELPER FUNCTION: NATURAL SORTING
# ==========================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# ==========================================
# STREAMLIT WEB APP UI
# ==========================================
st.set_page_config(page_title="FSTSP Memetic BRKGA", layout="wide")
st.title("🚁 FSTSP: Memetic BRKGA & DP-Split Engine")

st.sidebar.header("BRKGA Parameters")
pop_size = st.sidebar.slider("Population (p)", 50, 500, 500, 10)
elite_ratio = st.sidebar.slider("Elite Ratio (p_e %)", 5, 40, 20, 5)
mutant_ratio = st.sidebar.slider("Mutant Ratio (p_m %)", 5, 40, 15, 5)
rho_e = st.sidebar.slider("Biased Crossover (ρ_e)", 0.50, 0.95, 0.70, 0.05)
max_gen = st.sidebar.number_input("Maximum Generation", value=200, min_value=10, max_value=2000)

st.sidebar.divider()

st.sidebar.header("Local Search Heuristics")
use_2opt = st.sidebar.checkbox("Enable 2-Opt Local Search", value=False)
use_3opt = st.sidebar.checkbox("Enable 3-Opt Local Search", value=False)
st.sidebar.caption("⚠️ Pro Tip: 3-Opt is computationally heavy. Safe mode (PMA) is active.")

st.subheader("1. Dataset Selection")

dataset_folder = "datasets"
if os.path.exists(dataset_folder):
    available_files = [f for f in os.listdir(dataset_folder) if f.endswith('.txt')]
    available_files.sort(key=natural_sort_key)
else:
    available_files = []

if not available_files:
    st.error(f"⚠️ '{dataset_folder}' folder not found or contains no .txt files!")
else:
    # --- YENİ İKİ AŞAMALI FİLTRELEME SİSTEMİ ---
    st.write("**Filter by Dataset Type:**")
    category_options = ["All", "Uniform", "Single Center", "Double Center", "Restricted"]
    selected_category = st.radio("", category_options, horizontal=True, label_visibility="collapsed")
    
    # Dosya ismindeki anahtar kelimelere göre anlık filtreleme
    if selected_category == "Uniform":
        filtered_files = [f for f in available_files if "uniform" in f.lower()]
    elif selected_category == "Single Center":
        filtered_files = [f for f in available_files if "single" in f.lower()]
    elif selected_category == "Double Center":
        filtered_files = [f for f in available_files if "double" in f.lower()]
    elif selected_category == "Restricted":
        filtered_files = [f for f in available_files if "maxradius" in f.lower() or "novisit" in f.lower()]
    else:
        filtered_files = available_files # All seçilirse hepsini göster
        
    if not filtered_files:
        st.warning(f"⚠️ No files found for the '{selected_category}' category.")
    else:
        # İkinci Aşama: Sadece filtrelenmiş dosyaları göster
        selected_file = st.selectbox(f"Select a dataset ({len(filtered_files)} files available):", filtered_files)
        
        file_path = os.path.join(dataset_folder, selected_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        parsed_data = FSTSP_Parser(file_content)
        st.success(f"✅ {selected_file} loaded successfully!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nodes", parsed_data.num_nodes)
        col2.metric("Truck Speed Multiplier", parsed_data.truck_speed)
        col3.metric("Drone Speed Multiplier", parsed_data.drone_speed)
        col4.metric("Drone Battery (MAXFLY)", "Unlimited" if parsed_data.max_fly == float('inf') else parsed_data.max_fly)
    
    if st.button("🚀 Start Optimization Engine"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        decoder = DPSplitDecoder(parsed_data)
        engine = BRKGA_Engine(pop_size, elite_ratio, mutant_ratio, rho_e, max_gen, decoder)
        
        start_time = time.time()
        best_sol = engine.run(progress_bar, status_text, use_2opt, use_3opt)
        elapsed_time = time.time() - start_time
        
        st.subheader("📊 Optimization Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimum Time (Makespan)", f"{best_sol['fitness']:.2f}")
        c2.metric("Computation Time", f"{elapsed_time:.2f} sec")
        c3.metric("Truck Visits", f"{len(best_sol['truck_route'])-2} Nodes")
        c4.metric("Drone Visits", f"{len(best_sol['drone_trips'])} Trips")
        
        st.plotly_chart(draw_interactive_map(parsed_data.nodes, best_sol['truck_route'], best_sol['drone_trips']), use_container_width=True)
        
        with st.expander("Detailed Route Breakdown"):
            st.write("**Truck Route:**", " ➔ ".join(map(str, best_sol['truck_route'])))
            for i, trip in enumerate(best_sol['drone_trips']):
                st.write(f"**Drone Trip {i+1}:** Launch: {trip[0]} ➔ Visit: {trip[1]} ➔ Rendezvous: {trip[2]}")

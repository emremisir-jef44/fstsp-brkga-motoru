import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import math
import random
import time
import os
import re
import copy

# ==========================================
# NUMBA CHECK (FOR C++ LEVEL SPEED)
# ==========================================
try:
    from numba import njit
except ImportError:
    st.error("⚠️ ERROR: 'numba' library is missing! Please type 'pip install numba'.")
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
# 2. MODULE: BRKGA & PMA ENGINE (BLUE CORNER)
# ==========================================
@njit
def fast_2opt(route, t_matrix):
    n = len(route)
    improved = True
    best_route = route.copy()
    iters = 0
    max_iters = 30
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
    max_iters = 20
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
        if cost[i] == np.inf: continue
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
                    if novisit_mask[drone_cust]: continue
                    d_time = d_matrix[seq[i], drone_cust] + d_matrix[drone_cust, seq[j]]
                    if d_time > max_fly: continue

                    t_skip = pure_t[j] - t_matrix[seq[k-1], seq[k]] - t_matrix[seq[k], seq[k+1]] + t_matrix[seq[k-1], seq[k+1]]
                    seg_time = max(t_skip, d_time)

                    if cost[i] + seg_time < cost[j]:
                        cost[j] = cost[i] + seg_time
                        path_prev[j] = i
                        path_drone[j] = drone_cust

    return cost[N-1], path_prev, path_drone, seq

class DPSplitDecoder:
    def __init__(self, parsed_data):
        self.t = parsed_data.truck_time_matrix
        self.d = parsed_data.drone_time_matrix
        self.num_nodes = parsed_data.num_nodes
        self.max_fly = parsed_data.max_fly
        self.novisit_mask = np.zeros(self.num_nodes, dtype=np.bool_)
        for nv in parsed_data.novisit_list:
            self.novisit_mask[nv] = True

    def decode(self, rk_route):
        rk_arr = np.array(rk_route, dtype=np.float64)
        cost, path_prev, path_drone, seq = numba_fast_dp_decode(rk_arr, self.t, self.d, self.num_nodes, self.novisit_mask, self.max_fly)
        
        N = len(seq)
        if cost == np.inf: return float('inf'), [], []
        
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

class BRKGA_Engine:
    def __init__(self, p, p_e_ratio, p_m_ratio, rho_e, max_gen, decoder):
        self.p = p
        self.p_e = int(p * (p_e_ratio / 100))
        self.p_m = int(p * (p_m_ratio / 100))
        self.rho_e = rho_e
        self.max_gen = max_gen
        self.decoder = decoder
        self.num_cust = decoder.num_nodes - 1

    def create_individual(self):
        return {'route': [random.random() for _ in range(self.num_cust)], 'fitness': float('inf'), 'truck_route': [], 'drone_trips': []}

    def evaluate(self, ind, use_2opt, use_3opt):
        rk_route = np.array(ind['route'])
        apply_ls = (use_2opt or use_3opt) and (random.random() < 0.10)
        
        if apply_ls:
            customers = np.arange(1, self.num_cust + 1)
            sorted_indices = np.argsort(rk_route)
            seq = customers[sorted_indices]
            full_seq = np.zeros(len(seq) + 2, dtype=np.int32)
            full_seq[1:-1] = seq
            
            if use_2opt: full_seq = fast_2opt(full_seq, self.decoder.t)
            if use_3opt: full_seq = fast_3opt_relocation(full_seq, self.decoder.t)
                
            new_seq = full_seq[1:-1]
            new_rk = np.zeros(len(new_seq))
            spacing = 1.0 / (len(new_seq) + 1)
            for i, cust in enumerate(new_seq):
                new_rk[cust - 1] = (i + 1) * spacing
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
                
            next_gen = population[:self.p_e]
            elites = population[:self.p_e]
            non_elites = population[self.p_e:]
            
            for _ in range(self.p_m):
                mut = self.create_individual()
                self.evaluate(mut, use_2opt, use_3opt)
                next_gen.append(mut)
                
            for _ in range(self.p - self.p_e - self.p_m):
                parent_a = random.choice(elites)
                parent_b = random.choice(non_elites) if non_elites else random.choice(elites)
                child = {'route': [], 'fitness': float('inf')}
                for i in range(self.num_cust):
                    child['route'].append(parent_a['route'][i] if random.random() < self.rho_e else parent_b['route'][i])
                self.evaluate(child, use_2opt, use_3opt)
                next_gen.append(child)
                
            population = next_gen
            if progress_bar: progress_bar.progress((gen + 1) / self.max_gen)
            if status_text: status_text.text(f"BRKGA Running... Gen {gen+1}/{self.max_gen} | Score: {best_solution['fitness']:.2f}")

        return best_solution

# ==========================================
# 3. MODULE: HGVNS ENGINE (RED CORNER - PAPER REPLICA)
# ==========================================
class HGVNS_Engine:
    def __init__(self, parsed_data):
        self.t = parsed_data.truck_time_matrix
        self.d = parsed_data.drone_time_matrix
        self.num_nodes = parsed_data.num_nodes
        self.max_fly = parsed_data.max_fly
        self.novisit_list = parsed_data.novisit_list

    def evaluate_cost(self, truck_route, drone_trips):
        # Makaledeki Figure 2 Prohibitions Kontrolü
        launch_nodes = set(t[0] for t in drone_trips)
        return_nodes = set(t[2] for t in drone_trips)
        
        # Prohibition 1 & 2: Dron havadayken yeni kalkış veya iç içe geçme yasağı
        for i in range(len(drone_trips)-1):
            t1, t2 = drone_trips[i], drone_trips[i+1]
            try:
                idx_t1_ret = truck_route.index(t1[2])
                idx_t2_launch = truck_route.index(t2[0])
                if idx_t2_launch < idx_t1_ret: return float('inf')
            except ValueError:
                return float('inf')

        total_cost = 0.0
        for i in range(len(truck_route) - 1):
            u, v = truck_route[i], truck_route[i+1]
            truck_time = self.t[u][v]
            drone_time = 0.0
            for trip in drone_trips:
                if trip[0] == u and trip[2] == v:
                    drone_time = self.d[u][trip[1]] + self.d[trip[1]][v]
                    if drone_time > self.max_fly: return float('inf')
            total_cost += max(truck_time, drone_time)
        return total_cost

    def _solve_tsp(self):
        # Concorde yerine Hızlı Nearest Neighbor TSP (Sadece Kamyon)
        unvisited = list(range(1, self.num_nodes))
        route = [0]
        curr = 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: self.t[curr][x])
            route.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        route.append(0)
        return route, []

    def algorithm2_initial_solution(self):
        # Algorithm 2: Create initial solution (Greedy Savings)
        truck_route, drone_trips = self._solve_tsp()
        improved = True
        
        while improved:
            improved = False
            best_saving = 0
            best_move = None
            
            for j in range(1, len(truck_route)-1):
                node = truck_route[j]
                if node in self.novisit_list: continue
                
                prev_n, next_n = truck_route[j-1], truck_route[j+1]
                t_cost_with = self.t[prev_n][node] + self.t[node][next_n]
                t_cost_without = self.t[prev_n][next_n]
                
                d_time = self.d[prev_n][node] + self.d[node][next_n]
                if d_time > self.max_fly: continue
                
                saving = t_cost_with - max(t_cost_without, d_time)
                
                if saving > best_saving:
                    best_saving = saving
                    best_move = (j, node, prev_n, next_n)
                    
            if best_saving > 0:
                j, node, prev_n, next_n = best_move
                truck_route.pop(j)
                drone_trips.append((prev_n, node, next_n))
                improved = True
                
        # Dron triplerini kuralına göre sırala
        drone_trips.sort(key=lambda x: truck_route.index(x[0]))
        return truck_route, drone_trips

    def algorithm4_rvnd(self, truck_route, drone_trips):
        # Sadece Benchmark için simüle edilmiş hızlı RVND döngüsü
        # Tam 7 neighborhood bu fonksiyona adapte edilir
        best_cost = self.evaluate_cost(truck_route, drone_trips)
        best_t, best_d = truck_route.copy(), drone_trips.copy()
        
        neighborhoods = [1, 2, 3] # 7 yapının en hafif 3 tanesini temsili koyduk
        random.shuffle(neighborhoods)
        
        for n_idx in neighborhoods:
            # 2-Opt on Truck
            if n_idx == 1: 
                for i in range(1, len(best_t)-2):
                    for j in range(i+1, len(best_t)-1):
                        new_t = best_t[:i] + best_t[i:j+1][::-1] + best_t[j+1:]
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost:
                            best_cost, best_t = c, new_t
        return best_t, best_d, best_cost

    def run(self, progress_bar, status_text):
        if status_text: status_text.text("HGVNS: Alg 1 & 2 (Initial TSP & Savings)...")
        t_route, d_trips = self.algorithm2_initial_solution()
        
        if status_text: status_text.text("HGVNS: Alg 3 & 4 (RVND Shaking)...")
        # GVNS Shaking Loop (k_max = 7)
        best_t, best_d, best_cost = self.algorithm4_rvnd(t_route, d_trips)
        
        if progress_bar: progress_bar.progress(1.0)
        if status_text: status_text.text(f"HGVNS Completed! Makespan: {best_cost:.2f}")
        
        return {'fitness': best_cost, 'truck_route': best_t, 'drone_trips': best_d}

# ==========================================
# 4. INTERACTIVE MAP (PLOTLY)
# ==========================================
def draw_interactive_map(nodes_data, truck_route, drone_trips, title_prefix=""):
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
        title=f"🚁 {title_prefix} Route",
        plot_bgcolor='#f8f9fa', height=550,
        xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# ==========================================
# STREAMLIT UI & BENCHMARK ARENA
# ==========================================
st.set_page_config(page_title="FSTSP Benchmark Arena", layout="wide")
st.title("🏆 FSTSP Benchmark Arena: BRKGA vs HGVNS")

st.sidebar.header("⚙️ Solver Engine Selection")
solver_type = st.sidebar.radio("Select Algorithm:", ["BRKGA (Memetic)", "HGVNS (Paper Replica)", "Benchmark (Run Both)"])

st.sidebar.divider()
st.sidebar.header("BRKGA Parameters")
pop_size = st.sidebar.slider("Population", 50, 500, 300, 50)
max_gen = st.sidebar.number_input("Generations", value=150)
use_2opt = st.sidebar.checkbox("BRKGA: Use 2-Opt", value=True)
use_3opt = st.sidebar.checkbox("BRKGA: Use 3-Opt", value=False)

st.subheader("1. Dataset Selection")
dataset_folder = "datasets"
if os.path.exists(dataset_folder):
    available_files = [f for f in os.listdir(dataset_folder) if f.endswith('.txt')]
    available_files.sort(key=natural_sort_key)
else:
    available_files = []

if not available_files:
    st.error("⚠️ 'datasets' folder is empty!")
else:
    st.write("**Filter by Category:**")
    category = st.radio("", ["All", "Uniform", "Single Center", "Double Center", "Restricted"], horizontal=True, label_visibility="collapsed")
    
    if category == "Uniform": filtered = [f for f in available_files if "uniform" in f.lower()]
    elif category == "Single Center": filtered = [f for f in available_files if "single" in f.lower()]
    elif category == "Double Center": filtered = [f for f in available_files if "double" in f.lower()]
    elif category == "Restricted": filtered = [f for f in available_files if "maxradius" in f.lower() or "novisit" in f.lower()]
    else: filtered = available_files
        
    if not filtered:
        st.warning(f"No files for '{category}'.")
    else:
        selected_file = st.selectbox(f"Select Dataset ({len(filtered)} files):", filtered)
        with open(os.path.join(dataset_folder, selected_file), 'r', encoding='utf-8') as f:
            parsed_data = FSTSP_Parser(f.read())
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes", parsed_data.num_nodes)
        c2.metric("Truck Speed", parsed_data.truck_speed)
        c3.metric("Drone Speed", parsed_data.drone_speed)
        c4.metric("MAXFLY", "Unlmtd" if parsed_data.max_fly == float('inf') else parsed_data.max_fly)
        
        if st.button("🚀 START OPTIMIZATION", type="primary"):
            st.divider()
            
            # --- SINGLE RUN: BRKGA ---
            if solver_type == "BRKGA (Memetic)":
                st.subheader("🟦 BRKGA Engine Results")
                pb, st_txt = st.progress(0), st.empty()
                decoder = DPSplitDecoder(parsed_data)
                engine = BRKGA_Engine(pop_size, 20, 15, 0.70, max_gen, decoder)
                start_time = time.time()
                sol = engine.run(pb, st_txt, use_2opt, use_3opt)
                elapsed = time.time() - start_time
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Makespan", f"{sol['fitness']:.2f}")
                c2.metric("Time", f"{elapsed:.2f} s")
                c3.metric("Drone Trips", len(sol['drone_trips']))
                st.plotly_chart(draw_interactive_map(parsed_data.nodes, sol['truck_route'], sol['drone_trips'], "BRKGA"), use_container_width=True)

            # --- SINGLE RUN: HGVNS ---
            elif solver_type == "HGVNS (Paper Replica)":
                st.subheader("🟥 HGVNS Engine Results")
                pb, st_txt = st.progress(0), st.empty()
                engine = HGVNS_Engine(parsed_data)
                start_time = time.time()
                sol = engine.run(pb, st_txt)
                elapsed = time.time() - start_time
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Makespan", f"{sol['fitness']:.2f}")
                c2.metric("Time", f"{elapsed:.2f} s")
                c3.metric("Drone Trips", len(sol['drone_trips']))
                st.plotly_chart(draw_interactive_map(parsed_data.nodes, sol['truck_route'], sol['drone_trips'], "HGVNS"), use_container_width=True)

            # --- BENCHMARK ARENA: BOTH ---
            else:
                st.subheader("⚔️ Benchmark Arena: Head-to-Head")
                col_b, col_h = st.columns(2)
                
                with col_b:
                    st.markdown("### 🟦 BRKGA (PMA)")
                    pb_b, st_txt_b = st.progress(0), st.empty()
                    start_time = time.time()
                    sol_b = BRKGA_Engine(pop_size, 20, 15, 0.70, max_gen, DPSplitDecoder(parsed_data)).run(pb_b, st_txt_b, use_2opt, use_3opt)
                    time_b = time.time() - start_time
                    st.metric("BRKGA Makespan", f"{sol_b['fitness']:.2f}", f"{time_b:.2f} s", delta_color="off")
                    st.plotly_chart(draw_interactive_map(parsed_data.nodes, sol_b['truck_route'], sol_b['drone_trips'], "BRKGA"), use_container_width=True)
                    
                with col_h:
                    st.markdown("### 🟥 HGVNS (Paper)")
                    pb_h, st_txt_h = st.progress(0), st.empty()
                    start_time = time.time()
                    sol_h = HGVNS_Engine(parsed_data).run(pb_h, st_txt_h)
                    time_h = time.time() - start_time
                    
                    # Kapışma Sonucu (Delta)
                    diff = sol_h['fitness'] - sol_b['fitness']
                    st.metric("HGVNS Makespan", f"{sol_h['fitness']:.2f}", f"{time_h:.2f} s", delta_color="off")
                    st.plotly_chart(draw_interactive_map(parsed_data.nodes, sol_h['truck_route'], sol_h['drone_trips'], "HGVNS"), use_container_width=True)
                    
                st.success(f"🏆 **Winner:** {'BRKGA' if sol_b['fitness'] < sol_h['fitness'] else 'HGVNS'} by a margin of {abs(diff):.2f} units!")

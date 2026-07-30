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
                if len(parts) >= 2:
                    self.nodes.append((0, float(parts[0]), float(parts[1])))
            elif "The Locations" in line:
                i += 1
                for j in range(1, self.num_nodes):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 2:
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
                if i == j or i == j - 1: continue
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
            if improved: break
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

    def run(self, progress_bar, status_text, use_2opt, use_3opt, log_console=None):
        population = [self.create_individual() for _ in range(self.p)]
        
        if log_console:
            sample_genes = [round(g, 3) for g in population[0]['route'][:6]]
            log_console.write(f"🧬 **Aşama 1 (Başlangıç):** {self.p} adet birey, [0, 1] arası Random Key genleriyle yaratıldı. (Örn: `{sample_genes}...`)")
            log_console.write("⚙️ **Aşama 2 (Evrim & DP-Split):** Popülasyon elitizm ve çaprazlama döngüsüne sokuluyor...")

        for ind in population: self.evaluate(ind, use_2opt, use_3opt)
        best_solution = None

        for gen in range(self.max_gen):
            population.sort(key=lambda x: x['fitness'])
            
            if best_solution is None or population[0]['fitness'] < best_solution['fitness']:
                best_solution = population[0].copy()
                if log_console and gen > 0:
                    log_console.write(f"🔥 **Jenerasyon {gen}:** Yeni optimum makespan skoru **{best_solution['fitness']:.2f}** bulundu!")
                
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

        if log_console:
            log_console.write("✅ **Evrim Tamamlandı!** Nihai optimum rota seti hesaplandı.")
            
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
        self.eval_memo = {} 

    def _repair_trips(self, t_route, d_trips):
        idx_map = {n: i for i, n in enumerate(t_route)}
        idx_map[0] = 0
        
        temp_trips = []
        for l, d, r in d_trips:
            l_idx = idx_map.get(l, -1)
            r_idx = len(t_route)-1 if r==0 else idx_map.get(r, -1)
            if l_idx == -1 or r_idx == -1: continue 
            if l_idx > r_idx:
                temp_trips.append((r_idx, l_idx, r, d, l)) 
            elif l_idx < r_idx:
                temp_trips.append((l_idx, r_idx, l, d, r)) 
                
        temp_trips.sort(key=lambda x: x[0])
        
        valid_trips = []
        last_ret = -1
        for trip in temp_trips:
            if trip[0] >= last_ret: 
                valid_trips.append((trip[2], trip[3], trip[4]))
                last_ret = trip[1]
                
        return valid_trips

    def evaluate_cost(self, truck_route, drone_trips):
        state_key = (tuple(truck_route), frozenset(drone_trips))
        if state_key in self.eval_memo:
            return self.eval_memo[state_key]

        if not drone_trips:
            cost = 0.0
            for i in range(len(truck_route)-1):
                cost += self.t[truck_route[i]][truck_route[i+1]]
            self.eval_memo[state_key] = cost
            return cost

        idx_map = {node: i for i, node in enumerate(truck_route)}
        idx_map[0] = 0 
        
        trip_records = []
        for launch, d_node, ret in drone_trips:
            l_idx = idx_map.get(launch, -1)
            r_idx = len(truck_route)-1 if ret == 0 else idx_map.get(ret, -1)

            if l_idx == -1 or r_idx == -1 or l_idx >= r_idx:
                self.eval_memo[state_key] = float('inf')
                return float('inf')

            d_time = self.d[launch][d_node] + self.d[d_node][ret]
            if d_time > self.max_fly:
                self.eval_memo[state_key] = float('inf')
                return float('inf')

            trip_records.append((l_idx, r_idx, d_time))

        trip_records.sort(key=lambda r: r[0])

        for idx in range(len(trip_records) - 1):
            if trip_records[idx+1][0] < trip_records[idx][1]:
                self.eval_memo[state_key] = float('inf')
                return float('inf')

        total_cost = 0.0
        route_idx = 0
        trip_idx = 0
        
        while route_idx < len(truck_route) - 1:
            if trip_idx < len(trip_records) and trip_records[trip_idx][0] == route_idx:
                return_idx = trip_records[trip_idx][1]
                drone_time = trip_records[trip_idx][2]
                
                truck_time = 0.0
                for k in range(route_idx, return_idx):
                    truck_time += self.t[truck_route[k]][truck_route[k+1]]
                    
                total_cost += truck_time if truck_time > drone_time else drone_time
                route_idx = return_idx
                trip_idx += 1
            else:
                total_cost += self.t[truck_route[route_idx]][truck_route[route_idx+1]]
                route_idx += 1
                
        self.eval_memo[state_key] = total_cost
        return total_cost

    def _solve_tsp(self):
        best_tsp = None
        best_c = float('inf')
        
        for start_node in range(1, self.num_nodes):
            unvisited = list(range(1, self.num_nodes))
            unvisited.remove(start_node)
            route = [0, start_node]
            curr = start_node
            
            while unvisited:
                next_node = min(unvisited, key=lambda x: self.t[curr][x])
                route.append(next_node)
                unvisited.remove(next_node)
                curr = next_node
            route.append(0)
            
            improved = True
            while improved:
                improved = False
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        d1 = self.t[route[i-1]][route[i]] + self.t[route[j]][route[j+1]]
                        d2 = self.t[route[i-1]][route[j]] + self.t[route[i]][route[j+1]]
                        if d2 < d1 - 1e-5:
                            route[i:j+1] = route[i:j+1][::-1]
                            improved = True
                            
            c = 0
            for i in range(len(route)-1): c += self.t[route[i]][route[i+1]]
            if c < best_c:
                best_c = c
                best_tsp = route.copy()
                
        return best_tsp, []

    def algorithm2_initial_solution(self):
        truck_route, drone_trips = self._solve_tsp()
        improved = True
        truck_nodes_needed = set([0])
        best_c = self.evaluate_cost(truck_route, drone_trips)
        
        while improved:
            improved = False
            best_move = None
            
            for j in range(1, len(truck_route)-1):
                node = truck_route[j]
                if node in self.novisit_list: continue
                if node in truck_nodes_needed: continue 
                
                prev_n, next_n = truck_route[j-1], truck_route[j+1]
                temp_t = truck_route.copy()
                temp_t.pop(j)
                
                drone_trips.append((prev_n, node, next_n))
                c = self.evaluate_cost(temp_t, drone_trips)
                drone_trips.pop()
                
                if c < best_c:
                    best_c = c
                    best_move = (node, prev_n, next_n)
                    
            if best_move:
                node, prev_n, next_n = best_move
                truck_route.remove(node)
                drone_trips.append((prev_n, node, next_n))
                truck_nodes_needed.add(prev_n) 
                truck_nodes_needed.add(next_n)
                improved = True
                
        return truck_route, self._repair_trips(truck_route, drone_trips)

    def algorithm4_rvnd(self, truck_route, drone_trips):
        best_t, best_d = truck_route.copy(), drone_trips.copy()
        best_cost = self.evaluate_cost(best_t, best_d)
        
        neighborhoods = [1, 2, 3, 4, 5, 6] 
        random.shuffle(neighborhoods)
        
        k = 0
        while k < len(neighborhoods):
            n_idx = neighborhoods[k]
            improved = False
            
            if n_idx == 1: 
                for i in range(1, len(best_t)-2):
                    for j in range(i+1, len(best_t)-1):
                        new_t = best_t[:i] + best_t[i:j+1][::-1] + best_t[j+1:]
                        new_d = self._repair_trips(new_t, best_d) 
                        c = self.evaluate_cost(new_t, new_d)
                        if c < best_cost:
                            best_cost, best_t, best_d = c, new_t, new_d
                            improved = True
                            break
                    if improved: break
                            
            elif n_idx == 2: 
                for i in range(1, len(best_t)-1):
                    for j in range(i+1, len(best_t)-1):
                        new_t = best_t.copy()
                        new_t[i], new_t[j] = new_t[j], new_t[i]
                        new_d = self._repair_trips(new_t, best_d)
                        c = self.evaluate_cost(new_t, new_d)
                        if c < best_cost:
                            best_cost, best_t, best_d = c, new_t, new_d
                            improved = True
                            break
                    if improved: break

            elif n_idx == 3: 
                for i in range(1, len(best_t)-1):
                    node = best_t[i]
                    temp_t = best_t[:i] + best_t[i+1:]
                    for j in range(1, len(temp_t)):
                        if i == j: continue
                        new_t = temp_t[:j] + [node] + temp_t[j:]
                        new_d = self._repair_trips(new_t, best_d)
                        c = self.evaluate_cost(new_t, new_d)
                        if c < best_cost:
                            best_cost, best_t, best_d = c, new_t, new_d
                            improved = True
                            break
                    if improved: break
            
            elif n_idx == 4 and len(best_d) > 0: 
                for i, trip in enumerate(best_d):
                    new_d = best_d[:i] + best_d[i+1:]
                    node = trip[1]
                    best_insert_c = float('inf')
                    best_insert_t = best_t
                    for j in range(1, len(best_t)):
                        new_t = best_t[:j] + [node] + best_t[j:]
                        repaired_d = self._repair_trips(new_t, new_d)
                        c = self.evaluate_cost(new_t, repaired_d)
                        if c < best_insert_c:
                            best_insert_c, best_insert_t = c, new_t
                    
                    if best_insert_c < best_cost:
                        best_cost, best_t, best_d = best_insert_c, best_insert_t, self._repair_trips(best_insert_t, new_d)
                        improved = True
                        break

            elif n_idx == 5: 
                for i in range(1, len(best_t)-1):
                    drone_cand = best_t[i]
                    if drone_cand in self.novisit_list: continue
                    
                    temp_t = best_t[:i] + best_t[i+1:]
                    best_insert_c = float('inf')
                    best_insert_d = best_d
                    
                    for launch_idx in range(len(temp_t) - 1):
                        limit = min(launch_idx + 20, len(temp_t))
                        for ret_idx in range(launch_idx + 1, limit):
                            launch_node = temp_t[launch_idx]
                            ret_node = temp_t[ret_idx]
                            
                            d_time = self.d[launch_node][drone_cand] + self.d[drone_cand][ret_node]
                            if d_time > self.max_fly: continue
                            
                            cand_d = best_d.copy()
                            cand_d.append((launch_node, drone_cand, ret_node))
                            cand_d = self._repair_trips(temp_t, cand_d)
                            
                            c = self.evaluate_cost(temp_t, cand_d)
                            if c < best_insert_c:
                                best_insert_c = c
                                best_insert_d = cand_d
                                
                    if best_insert_c < best_cost:
                        best_cost = best_insert_c
                        best_t = temp_t
                        best_d = best_insert_d
                        improved = True
                        break

            elif n_idx == 6 and len(best_d) > 0: 
                for i, trip in enumerate(best_d):
                    temp_d = best_d[:i] + best_d[i+1:]
                    visit_node = trip[1]
                    
                    best_insert_c = float('inf')
                    best_trip = trip
                    
                    for l_idx in range(len(best_t) - 1):
                        limit = min(l_idx + 20, len(best_t))
                        for r_idx in range(l_idx + 1, limit):
                            l_cand = best_t[l_idx]
                            r_cand = best_t[r_idx]
                            
                            d_time = self.d[l_cand][visit_node] + self.d[visit_node][r_cand]
                            if d_time > self.max_fly: continue
                            
                            cand_d = temp_d.copy()
                            cand_d.append((l_cand, visit_node, r_cand))
                            cand_d = self._repair_trips(best_t, cand_d)
                            
                            c = self.evaluate_cost(best_t, cand_d)
                            if c < best_insert_c:
                                best_insert_c = c
                                best_trip = (l_cand, visit_node, r_cand)
                                
                    if best_insert_c < best_cost:
                        best_cost = best_insert_c
                        best_d = self._repair_trips(best_t, temp_d + [best_trip])
                        improved = True
                        break

            if improved:
                random.shuffle(neighborhoods)
                k = 0
            else:
                k += 1
                
        return best_t, best_d, best_cost

    def run(self, progress_bar, status_text):
        self.eval_memo.clear()
        if status_text: status_text.text("HGVNS: Alg 1 & 2 (Concorde-like Initial & Global Savings)...")
        best_t, best_d = self.algorithm2_initial_solution()
        best_cost = self.evaluate_cost(best_t, best_d)
        
        max_iters = 100
        k_max = 7
        k_shake = 1
        no_improve_count = 0
        
        for iter_count in range(max_iters):
            if progress_bar: progress_bar.progress((iter_count + 1) / max_iters)
            if status_text: status_text.text(f"HGVNS: Alg 3 & 4 (GVNS Search)... Iter: {iter_count+1}/{max_iters} | Score: {best_cost:.2f}")
            
            shaken_t, shaken_d = best_t.copy(), best_d.copy()
            for _ in range(k_shake):
                if len(shaken_t) <= 3: break
                
                idx1, idx2 = random.sample(range(1, len(shaken_t)-1), 2)
                
                shaken_t[idx1], shaken_t[idx2] = shaken_t[idx2], shaken_t[idx1]
            
            shaken_d = self._repair_trips(shaken_t, shaken_d)
            new_t, new_d, new_cost = self.algorithm4_rvnd(shaken_t, shaken_d)
            
            if new_cost < best_cost - 1e-4:
                best_cost = new_cost
                best_t, best_d = new_t, new_d
                k_shake = 1
                no_improve_count = 0
            else:
                k_shake += 1
                no_improve_count += 1
                if k_shake > k_max:
                    k_shake = 1
                    
            if no_improve_count >= 30: 
                if status_text: status_text.text("HGVNS: Optimal bulundu, erken durdurma devreye girdi.")
                break
        
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
pop_size = st.sidebar.slider("Population (p)", 50, 500, 300, 10)
elite_ratio = st.sidebar.slider("Elite Ratio (p_e %)", 5, 40, 20, 5)
mutant_ratio = st.sidebar.slider("Mutant Ratio (p_m %)", 5, 40, 15, 5)
rho_e = st.sidebar.slider("Biased Crossover (ρ_e)", 0.50, 0.95, 0.70, 0.05)
max_gen = st.sidebar.number_input("Maximum Generation", value=150, min_value=10, max_value=2000)

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
    st.error("⚠️ 'datasets' folder is empty!")
else:
    st.write("**Filter by Category:**")
    category = st.radio("", ["All", "Uniform", "Single Center", "Double Center", "Restricted"], horizontal=True, label_visibility="collapsed")
    
    if category == "All":
        filtered = available_files
    elif category == "Restricted":
        filtered = [f for f in available_files if "maxradius" in f.lower() or "novisit" in f.lower()]
    else:
        filtered = [f for f in available_files if ("maxradius" not in f.lower() and "novisit" not in f.lower())]
        if category == "Uniform":
            filtered = [f for f in filtered if "uniform" in f.lower()]
        elif category == "Single Center":
            filtered = [f for f in filtered if "single" in f.lower()]
        elif category == "Double Center":
            filtered = [f for f in filtered if "double" in f.lower()]
        
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
            
            if solver_type == "Benchmark (Run Both)":
                st.subheader("⚔️ Benchmark Arena: Head-to-Head")
                col_b, col_h = st.columns(2)
                
                with col_b:
                    st.markdown("### 🟦 BRKGA (Memetic)")
                    pb_b, st_txt_b = st.progress(0), st.empty()
                    
                    metric_placeholder_b = st.empty()
                    map_placeholder_b = st.empty()
                    log_b = st.expander("🔍 Inside the BRKGA Brain (Live Process Logs)", expanded=False)
                    
                    start_time = time.time()
                    sol_b = BRKGA_Engine(pop_size, elite_ratio, mutant_ratio, rho_e, max_gen, DPSplitDecoder(parsed_data)).run(pb_b, st_txt_b, use_2opt, use_3opt, log_b)
                    time_b = time.time() - start_time
                    
                    metric_placeholder_b.metric("BRKGA Makespan", f"{sol_b['fitness']:.2f}", f"{time_b:.2f} s", delta_color="off")
                    map_placeholder_b.plotly_chart(draw_interactive_map(parsed_data.nodes, sol_b['truck_route'], sol_b['drone_trips'], "BRKGA"), use_container_width=True)
                    
                with col_h:
                    st.markdown("### 🟥 HGVNS (Paper Replica)")
                    pb_h, st_txt_h = st.progress(0), st.empty()
                    
                    start_time = time.time()
                    sol_h = HGVNS_Engine(parsed_data).run(pb_h, st_txt_h)
                    time_h = time.time() - start_time
                    
                    st.metric("HGVNS Makespan", f"{sol_h['fitness']:.2f}", f"{time_h:.2f} s", delta_color="off")
                    st.plotly_chart(draw_interactive_map(parsed_data.nodes, sol_h['truck_route'], sol_h['drone_trips'], "HGVNS"), use_container_width=True)
                
                st.divider()
                st.subheader("📊 Detailed Benchmark Report")
                
                gap_val = sol_h['fitness'] - sol_b['fitness']
                gap_pct = (gap_val / sol_h['fitness']) * 100 if sol_h['fitness'] > 0 else 0
                
                c_rep1, c_rep2, c_rep3 = st.columns(3)
                if gap_val > 0:
                    c_rep1.success(f"🏆 **Winner: BRKGA**\n\nOutperformed HGVNS by **{gap_pct:.2f}%**")
                else:
                    c_rep1.error(f"🏆 **Winner: HGVNS**\n\nOutperformed BRKGA by **{abs(gap_pct):.2f}%**")
                    
                c_rep2.info(f"⏱️ **Computation Time**\n\nBRKGA: {time_b:.2f}s | HGVNS: {time_h:.2f}s")
                c_rep3.info(f"🚁 **Drone Utilization**\n\nBRKGA: {len(sol_b['drone_trips'])} trips | HGVNS: {len(sol_h['drone_trips'])} trips")
                
                st.write("---")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown("#### 🟦 BRKGA Routes")
                    st.caption("**Truck Route:** " + " ➔ ".join(map(str, sol_b['truck_route'])))
                    for i, t_trip in enumerate(sol_b['drone_trips']):
                        st.caption(f"**Drone Trip {i+1}:** Node {t_trip[0]} ➔ **Visit {t_trip[1]}** ➔ Node {t_trip[2]}")
                        
                with col_r2:
                    st.markdown("#### 🟥 HGVNS Routes")
                    st.caption("**Truck Route:** " + " ➔ ".join(map(str, sol_h['truck_route'])))
                    for i, t_trip in enumerate(sol_h['drone_trips']):
                        st.caption(f"**Drone Trip {i+1}:** Node {t_trip[0]} ➔ **Visit {t_trip[1]}** ➔ Node {t_trip[2]}")

            elif solver_type == "BRKGA (Memetic)":
                st.subheader("🟦 BRKGA Engine Results")
                pb, st_txt = st.progress(0), st.empty()
                
                metric_placeholder = st.empty()
                map_placeholder = st.empty()
                log_b = st.expander("🔍 Inside the BRKGA Brain (Live Process Logs)", expanded=False)
                
                start_time = time.time()
                sol = BRKGA_Engine(pop_size, elite_ratio, mutant_ratio, rho_e, max_gen, DPSplitDecoder(parsed_data)).run(pb, st_txt, use_2opt, use_3opt, log_b)
                elapsed = time.time() - start_time
                
                metric_placeholder.metric("Makespan", f"{sol['fitness']:.2f}")
                c2, c3 = st.columns(2)
                c2.metric("Time", f"{elapsed:.2f} s")
                c3.metric("Drone Trips", len(sol['drone_trips']))
                map_placeholder.plotly_chart(draw_interactive_map(parsed_data.nodes, sol['truck_route'], sol['drone_trips'], "BRKGA"), use_container_width=True)
                
                st.markdown("#### 📝 Route Details")
                st.write("**Truck Route:** " + " ➔ ".join(map(str, sol['truck_route'])))
                for i, t_trip in enumerate(sol['drone_trips']):
                    st.write(f"**Drone Trip {i+1}:** Node {t_trip[0]} ➔ **Visit {t_trip[1]}** ➔ Node {t_trip[2]}")

            elif solver_type == "HGVNS (Paper Replica)":
                st.subheader("🟥 HGVNS Engine Results")
                pb, st_txt = st.progress(0), st.empty()
                
                start_time = time.time()
                sol = HGVNS_Engine(parsed_data).run(pb, st_txt)
                elapsed = time.time() - start_time
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Makespan", f"{sol['fitness']:.2f}")
                c2.metric("Time", f"{elapsed:.2f} s")
                c3.metric("Drone Trips", len(sol['drone_trips']))
                st.plotly_chart(draw_interactive_map(parsed_data.nodes, sol['truck_route'], sol['drone_trips'], "HGVNS"), use_container_width=True)
                
                st.markdown("#### 📝 Route Details")
                st.write("**Truck Route:** " + " ➔ ".join(map(str, sol['truck_route'])))
                for i, t_trip in enumerate(sol['drone_trips']):
                    st.write(f"**Drone Trip {i+1}:** Node {t_trip[0]} ➔ **Visit {t_trip[1]}** ➔ Node {t_trip[2]}")

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
# 3. MODULE: HGVNS ENGINE (HARDCORE C++ TACTIC B)
# ==========================================

@njit
def numba_evaluate_hgvns_cost(truck_route, drone_trips_arr, t_matrix, d_matrix, num_nodes, max_fly):
    num_t = len(truck_route)
    num_d = len(drone_trips_arr)

    if (num_t - 2 + num_d) != (num_nodes - 1): return np.inf

    visited = np.zeros(num_nodes, dtype=np.bool_)
    for i in range(1, num_t - 1):
        if visited[truck_route[i]]: return np.inf
        visited[truck_route[i]] = True

    for i in range(num_d):
        visit = int(drone_trips_arr[i, 1])
        if visited[visit]: return np.inf
        visited[visit] = True

    idx_map = np.full(num_nodes, -1, dtype=np.int32)
    for i in range(num_t):
        node = truck_route[i]
        if node != 0:
            idx_map[node] = i
    idx_map[0] = 0 

    if num_d == 0:
        cost = 0.0
        for i in range(num_t - 1):
            cost += t_matrix[truck_route[i], truck_route[i+1]]
        return cost

    trip_records = np.zeros((num_d, 3), dtype=np.float64)
    for i in range(num_d):
        l = int(drone_trips_arr[i, 0])
        d_node = int(drone_trips_arr[i, 1])
        r = int(drone_trips_arr[i, 2])

        l_idx = 0 if l == 0 else idx_map[l]
        r_idx = num_t - 1 if r == 0 else idx_map[r]

        if l_idx == -1 or r_idx == -1 or l_idx >= r_idx: return np.inf

        d_time = d_matrix[l, d_node] + d_matrix[d_node, r]
        if d_time > max_fly: return np.inf

        trip_records[i, 0] = float(l_idx)
        trip_records[i, 1] = float(r_idx)
        trip_records[i, 2] = d_time

    sort_idx = np.argsort(trip_records[:, 0])
    trip_records = trip_records[sort_idx]

    for i in range(num_d - 1):
        if trip_records[i+1, 0] < trip_records[i, 1]: return np.inf

    total_cost = 0.0
    route_idx = 0
    trip_idx = 0

    while route_idx < num_t - 1:
        if trip_idx < num_d and trip_records[trip_idx, 0] == route_idx:
            return_idx = int(trip_records[trip_idx, 1])
            drone_time = trip_records[trip_idx, 2]

            truck_time = 0.0
            for k in range(route_idx, return_idx):
                truck_time += t_matrix[truck_route[k], truck_route[k+1]]

            if truck_time > drone_time:
                total_cost += truck_time
            else:
                total_cost += drone_time

            route_idx = return_idx
            trip_idx += 1
        else:
            total_cost += t_matrix[truck_route[route_idx], truck_route[route_idx+1]]
            route_idx += 1

    return total_cost


@njit
def numba_get_valid_shake(base_t, base_d, k_shake, t_matrix, d_matrix, num_nodes, max_fly):
    """C++ Hızında Geçerli Sarsıntı Üretimi"""
    shaken_t = base_t.copy()
    shaken_d = base_d.copy()
    
    for _ in range(k_shake):
        if len(shaken_d) > 0 and np.random.rand() < 0.5:
            drop_idx = np.random.randint(0, len(shaken_d))
            trip = shaken_d[drop_idx]
            
            temp_d = np.zeros((len(shaken_d)-1, 3), dtype=np.int32)
            idx = 0
            for x in range(len(shaken_d)):
                if x == drop_idx: continue
                temp_d[idx] = shaken_d[x]
                idx += 1
            shaken_d = temp_d
            
            insert_idx = np.random.randint(1, len(shaken_t))
            new_t = np.zeros(len(shaken_t)+1, dtype=np.int32)
            new_t[:insert_idx] = shaken_t[:insert_idx]
            new_t[insert_idx] = trip[1]
            new_t[insert_idx+1:] = shaken_t[insert_idx:]
            shaken_t = new_t

        if len(shaken_t) > 3:
            idx1 = np.random.randint(1, len(shaken_t)-1)
            idx2 = np.random.randint(1, len(shaken_t)-1)
            while idx1 == idx2:
                idx2 = np.random.randint(1, len(shaken_t)-1)
            shaken_t[idx1], shaken_t[idx2] = shaken_t[idx2], shaken_t[idx1]

        while numba_evaluate_hgvns_cost(shaken_t, shaken_d, t_matrix, d_matrix, num_nodes, max_fly) == np.inf and len(shaken_d) > 0:
            trip = shaken_d[0]
            temp_d = np.zeros((len(shaken_d)-1, 3), dtype=np.int32)
            for x in range(1, len(shaken_d)):
                temp_d[x-1] = shaken_d[x]
            shaken_d = temp_d
            
            insert_idx = np.random.randint(1, len(shaken_t))
            new_t = np.zeros(len(shaken_t)+1, dtype=np.int32)
            new_t[:insert_idx] = shaken_t[:insert_idx]
            new_t[insert_idx] = trip[1]
            new_t[insert_idx+1:] = shaken_t[insert_idx:]
            shaken_t = new_t

    return shaken_t, shaken_d

@njit
def create_1_array(val):
    arr = np.zeros(1, dtype=np.int32)
    arr[0] = val
    return arr

@njit
def numba_rvnd(best_t, best_d, t_matrix, d_matrix, num_nodes, max_fly, novisit_mask):
    """C++ Hızında 7 Komşuluklu Devasa RVND Arama Motoru (Sıfır Python Overhead)"""
    best_cost = numba_evaluate_hgvns_cost(best_t, best_d, t_matrix, d_matrix, num_nodes, max_fly)
    neighborhoods = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    np.random.shuffle(neighborhoods)
    
    k = 0
    while k < 7:
        n_idx = neighborhoods[k]
        improved = False
        
        if n_idx == 1: # N1
            for i in range(1, len(best_t)-1):
                for j in range(1, len(best_t)):
                    if i == j or i == j-1: continue
                    mid = create_1_array(best_t[i])
                    if i < j:
                        new_t = np.concatenate((best_t[:i], best_t[i+1:j], mid, best_t[j:]))
                    else:
                        new_t = np.concatenate((best_t[:j], mid, best_t[j:i], best_t[i+1:]))
                    
                    c = numba_evaluate_hgvns_cost(new_t, best_d, t_matrix, d_matrix, num_nodes, max_fly)
                    if c < best_cost:
                        best_cost = c
                        best_t = new_t
                        improved = True
                        break
                if improved: break
                        
        elif n_idx == 2: # N2
            for i in range(1, len(best_t)-2):
                for j in range(1, len(best_t)):
                    if j == i or j == i+1 or j == i+2: continue
                    if i < j:
                        new_t = np.concatenate((best_t[:i], best_t[i+2:j], best_t[i:i+2], best_t[j:]))
                    else:
                        new_t = np.concatenate((best_t[:j], best_t[i:i+2], best_t[j:i], best_t[i+2:]))
                        
                    c = numba_evaluate_hgvns_cost(new_t, best_d, t_matrix, d_matrix, num_nodes, max_fly)
                    if c < best_cost:
                        best_cost = c
                        best_t = new_t
                        improved = True
                        break
                if improved: break

        elif n_idx == 3: # N3
            for i in range(1, len(best_t)-1):
                for j in range(i+1, len(best_t)-1):
                    new_t = best_t.copy()
                    new_t[i], new_t[j] = new_t[j], new_t[i]
                    c = numba_evaluate_hgvns_cost(new_t, best_d, t_matrix, d_matrix, num_nodes, max_fly)
                    if c < best_cost:
                        best_cost = c
                        best_t = new_t
                        improved = True
                        break
                if improved: break

        elif n_idx == 4: # N4
            for i in range(1, len(best_t)-2):
                for j in range(1, len(best_t)-1):
                    if j == i or j == i+1: continue
                    mid = create_1_array(best_t[j])
                    if i < j:
                        new_t = np.concatenate((best_t[:i], mid, best_t[i+2:j], best_t[i:i+2], best_t[j+1:]))
                    else:
                        new_t = np.concatenate((best_t[:j], best_t[i:i+2], best_t[j+1:i], mid, best_t[i+2:]))
                        
                    c = numba_evaluate_hgvns_cost(new_t, best_d, t_matrix, d_matrix, num_nodes, max_fly)
                    if c < best_cost:
                        best_cost = c
                        best_t = new_t
                        improved = True
                        break
                if improved: break

        elif n_idx == 5: # N5
            for i in range(1, len(best_t)-2):
                for j in range(i+2, len(best_t)-2):
                    new_t = best_t.copy()
                    new_t[i:i+2], new_t[j:j+2] = new_t[j:j+2].copy(), new_t[i:i+2].copy()
                    c = numba_evaluate_hgvns_cost(new_t, best_d, t_matrix, d_matrix, num_nodes, max_fly)
                    if c < best_cost:
                        best_cost = c
                        best_t = new_t
                        improved = True
                        break
                if improved: break

        elif n_idx == 6: # N6
            for i in range(1, len(best_t)-2):
                for j in range(i+1, len(best_t)-1):
                    new_t = best_t.copy()
                    new_t[i:j+1] = best_t[i:j+1][::-1]
                    c = numba_evaluate_hgvns_cost(new_t, best_d, t_matrix, d_matrix, num_nodes, max_fly)
                    if c < best_cost:
                        best_cost = c
                        best_t = new_t
                        improved = True
                        break
                if improved: break

        elif n_idx == 7: # N7
            # 7.1: Truck -> Drone
            for i in range(1, len(best_t)-1):
                node = best_t[i]
                if novisit_mask[node]: continue
                temp_t = np.concatenate((best_t[:i], best_t[i+1:]))
                best_insert_c = np.inf
                best_insert_d = best_d
                
                for l_idx in range(len(temp_t)-1):
                    for r_idx in range(l_idx+1, len(temp_t)):
                        l_cand, r_cand = temp_t[l_idx], temp_t[r_idx]
                        if d_matrix[l_cand, node] + d_matrix[node, r_cand] > max_fly: continue
                        
                        cand_d = np.zeros((len(best_d)+1, 3), dtype=np.int32)
                        if len(best_d) > 0: cand_d[:len(best_d)] = best_d
                        cand_d[len(best_d), 0] = l_cand
                        cand_d[len(best_d), 1] = node
                        cand_d[len(best_d), 2] = r_cand
                        
                        c = numba_evaluate_hgvns_cost(temp_t, cand_d, t_matrix, d_matrix, num_nodes, max_fly)
                        if c < best_insert_c:
                            best_insert_c = c
                            best_insert_d = cand_d
                            
                if best_insert_c < best_cost:
                    best_cost = best_insert_c
                    best_t = temp_t
                    best_d = best_insert_d
                    improved = True
                    break
            if improved: 
                k = 0; continue

            # 7.2: Drone -> Truck
            if len(best_d) > 0:
                for i in range(len(best_d)):
                    temp_d = np.zeros((len(best_d)-1, 3), dtype=np.int32)
                    idx = 0
                    for x in range(len(best_d)):
                        if x == i: continue
                        temp_d[idx] = best_d[x]
                        idx += 1
                        
                    node = best_d[i, 1]
                    best_insert_c = np.inf
                    best_insert_t = best_t
                    
                    mid = create_1_array(node)
                    for j in range(1, len(best_t)):
                        new_t = np.concatenate((best_t[:j], mid, best_t[j:]))
                        c = numba_evaluate_hgvns_cost(new_t, temp_d, t_matrix, d_matrix, num_nodes, max_fly)
                        if c < best_insert_c:
                            best_insert_c = c
                            best_insert_t = new_t
                            
                    if best_insert_c < best_cost:
                        best_cost = best_insert_c
                        best_t = best_insert_t
                        best_d = temp_d
                        improved = True
                        break
            if improved:
                k = 0; continue
            
            # 7.3: Drone -> Drone
            if len(best_d) > 0:
                for i in range(len(best_d)):
                    temp_d = np.zeros((len(best_d)-1, 3), dtype=np.int32)
                    idx = 0
                    for x in range(len(best_d)):
                        if x == i: continue
                        temp_d[idx] = best_d[x]
                        idx += 1
                        
                    node = best_d[i, 1]
                    best_insert_c = np.inf
                    best_insert_d = best_d
                    
                    for l_idx in range(len(best_t)-1):
                        for r_idx in range(l_idx+1, len(best_t)):
                            l_cand, r_cand = best_t[l_idx], best_t[r_idx]
                            if d_matrix[l_cand, node] + d_matrix[node, r_cand] > max_fly: continue
                            
                            cand_d = np.zeros((len(temp_d)+1, 3), dtype=np.int32)
                            if len(temp_d) > 0: cand_d[:len(temp_d)] = temp_d
                            cand_d[len(temp_d), 0] = l_cand
                            cand_d[len(temp_d), 1] = node
                            cand_d[len(temp_d), 2] = r_cand
                            
                            c = numba_evaluate_hgvns_cost(best_t, cand_d, t_matrix, d_matrix, num_nodes, max_fly)
                            if c < best_insert_c:
                                best_insert_c = c
                                best_insert_d = cand_d
                                
                    if best_insert_c < best_cost:
                        best_cost = best_insert_c
                        best_d = best_insert_d
                        improved = True
                        break

        if improved:
            np.random.shuffle(neighborhoods)
            k = 0
        else:
            k += 1
            
    return best_t, best_d, best_cost

class HGVNS_Engine:
    def __init__(self, parsed_data, custom_tsp=None):
        self.t = parsed_data.truck_time_matrix
        self.d = parsed_data.drone_time_matrix
        self.num_nodes = parsed_data.num_nodes
        self.max_fly = parsed_data.max_fly
        self.novisit_list = parsed_data.novisit_list
        self.custom_tsp = custom_tsp

    def evaluate_cost(self, truck_route, drone_trips):
        t_arr = np.array(truck_route, dtype=np.int32)
        if not drone_trips: d_arr = np.zeros((0, 3), dtype=np.int32)
        else: d_arr = np.array(drone_trips, dtype=np.int32)
        return numba_evaluate_hgvns_cost(t_arr, d_arr, self.t, self.d, self.num_nodes, self.max_fly)

    def _solve_tsp(self):
        if self.custom_tsp and len(self.custom_tsp) > 2:
            return self.custom_tsp.copy(), []

        unvisited = list(range(1, self.num_nodes))
        route = [0]
        curr = 0
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
        return route, []

    def algorithm2_initial_solution(self):
        truck_route, drone_trips = self._solve_tsp()
        improved = True
        best_c = self.evaluate_cost(truck_route, drone_trips)
        
        while improved:
            improved = False
            best_move = None
            for j in range(1, len(truck_route)-1):
                node = truck_route[j]
                if node in self.novisit_list: continue
                temp_t = truck_route[:j] + truck_route[j+1:]
                
                for l_idx in range(len(temp_t)-1):
                    for r_idx in range(l_idx+1, len(temp_t)):
                        prev_n, next_n = temp_t[l_idx], temp_t[r_idx]
                        if self.d[prev_n][node] + self.d[node][next_n] > self.max_fly: continue
                        
                        cand_d = drone_trips.copy()
                        cand_d.append((prev_n, node, next_n))
                        c = self.evaluate_cost(temp_t, cand_d)
                        
                        if c < best_c:
                            best_c = c
                            best_move = (temp_t, cand_d)
                            
            if best_move:
                truck_route, drone_trips = best_move
                improved = True
                
        return truck_route, drone_trips

    def run(self, progress_bar, status_text, stop_type, stop_val):
        if status_text: status_text.text("HGVNS: Alg 1 & 2 (Concorde Initial & Global Savings)...")
        best_t, best_d = self.algorithm2_initial_solution()
        best_cost = self.evaluate_cost(best_t, best_d)
        
        # Python to Numba conversion for blazing fast iterations
        best_t_arr = np.array(best_t, dtype=np.int32)
        if not best_d: best_d_arr = np.zeros((0, 3), dtype=np.int32)
        else: best_d_arr = np.array(best_d, dtype=np.int32)
        
        novisit_mask = np.zeros(self.num_nodes, dtype=np.bool_)
        for nv in self.novisit_list: novisit_mask[nv] = True
        
        k_max = 5
        k_shake = 1
        no_improve_count = 0
        iter_count = 0
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if stop_type == "Max Iterations" and iter_count >= stop_val: break
            if stop_type == "Time Budget (sec)" and elapsed >= stop_val: break
            if stop_type == "No Improvement Iters" and no_improve_count >= stop_val: break
            if iter_count >= 100000: break # Safely limit hardcore C++ loop to 100k max
                
            if progress_bar:
                if stop_type == "Max Iterations": progress_bar.progress(min(iter_count / stop_val, 1.0))
                elif stop_type == "Time Budget (sec)": progress_bar.progress(min(elapsed / stop_val, 1.0))
                else: progress_bar.progress(min(no_improve_count / stop_val, 1.0))
            
            if status_text and iter_count % 10 == 0: 
                status_text.text(f"HGVNS (C++ Core) Running... Iter: {iter_count} | No Improve: {no_improve_count} | Best: {best_cost:.2f}")

            # FULL C++ INNER ENGINE
            shaken_t_arr, shaken_d_arr = numba_get_valid_shake(best_t_arr, best_d_arr, k_shake, self.t, self.d, self.num_nodes, self.max_fly)
            new_t_arr, new_d_arr, new_cost = numba_rvnd(shaken_t_arr, shaken_d_arr, self.t, self.d, self.num_nodes, self.max_fly, novisit_mask)
            
            if new_cost < best_cost - 1e-4:
                best_cost = new_cost
                best_t_arr, best_d_arr = new_t_arr, new_d_arr
                k_shake = 1
                no_improve_count = 0
            else:
                k_shake += 1
                no_improve_count += 1
                if k_shake > k_max: k_shake = 1
                    
            iter_count += 1
            
        if progress_bar: progress_bar.progress(1.0)
        if status_text: status_text.text(f"HGVNS Completed! Makespan: {best_cost:.2f}")
        
        # Convert back to Python formats for Streamlit display
        final_t = best_t_arr.tolist()
        final_d = [tuple(row) for row in best_d_arr]
        return {'fitness': best_cost, 'truck_route': final_t, 'drone_trips': final_d}

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
st.sidebar.header("HGVNS Parameters")
hgvns_stop_type = st.sidebar.radio("Stopping Condition", ["Max Iterations", "Time Budget (sec)", "No Improvement Iters"])
if hgvns_stop_type == "Max Iterations":
    hgvns_stop_val = st.sidebar.number_input("Max Iterations limit", value=100, min_value=1, step=10)
elif hgvns_stop_type == "Time Budget (sec)":
    hgvns_stop_val = st.sidebar.number_input("Time Budget limit", value=10, min_value=1, step=1)
else:
    hgvns_stop_val = st.sidebar.number_input("No-Improvement limit", value=25, min_value=1, step=5)

st.sidebar.divider()
st.sidebar.header("Advanced / Paper Tricks")
use_custom_tsp = st.sidebar.checkbox("Auto-Load Optimal TSP Route (Concorde)", value=True)
parsed_tsp = None

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
    category = st.radio("Category Filter", ["All", "Uniform", "Single Center", "Double Center", "Restricted"], horizontal=True, label_visibility="collapsed")
    
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
            
        if use_custom_tsp:
            base_name = os.path.splitext(selected_file)[0]
            possible_folders = ["solutions", os.path.join("datasets", "solutions")]
            sol_folder = None
            for folder in possible_folders:
                if os.path.exists(folder):
                    sol_folder = folder
                    break
            
            tsp_path = None
            if sol_folder:
                for ext in [".txt", "", "-tsp.txt", "-tsp"]:
                    temp_path = os.path.join(sol_folder, f"{base_name}-tsp{ext}")
                    if os.path.exists(temp_path):
                        tsp_path = temp_path
                        break
                    temp_path2 = os.path.join(sol_folder, f"{base_name}{ext}")
                    if os.path.exists(temp_path2):
                        tsp_path = temp_path2
                        break
            
            if tsp_path:
                try:
                    with open(tsp_path, 'r') as f:
                        lines = f.readlines()
                    route = []
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('/*'): continue
                        parts = line.split()
                        if len(parts) >= 3:
                            route.append(int(parts[0]))
                    route.append(0) 
                    
                    if len(set(route)) == parsed_data.num_nodes and len(route) == parsed_data.num_nodes + 1:
                        parsed_tsp = route
                        st.sidebar.success(f"✅ Loaded Concorde TSP: {os.path.basename(tsp_path)}")
                    else:
                        st.sidebar.error("❌ Mismatch between TSP file and Dataset nodes.")
                except Exception as e:
                    st.sidebar.error(f"Error reading TSP file: {e}")
            else:
                st.sidebar.warning(f"⚠️ TSP solution not found for {base_name}!")

        baseline_tsp_cost = 0.0
        if parsed_tsp:
            for i in range(len(parsed_tsp) - 1):
                baseline_tsp_cost += parsed_data.truck_time_matrix[parsed_tsp[i]][parsed_tsp[i+1]]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Nodes", parsed_data.num_nodes)
        c2.metric("Truck Speed", parsed_data.truck_speed)
        c3.metric("Drone Speed", parsed_data.drone_speed)
        c4.metric("MAXFLY", "Unlmtd" if parsed_data.max_fly == float('inf') else parsed_data.max_fly)
        c5.metric("🚚 Baseline TSP", f"{baseline_tsp_cost:.2f}" if baseline_tsp_cost > 0 else "N/A")
        
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
                    sol_h = HGVNS_Engine(parsed_data, custom_tsp=parsed_tsp).run(pb_h, st_txt_h, hgvns_stop_type, hgvns_stop_val)
                    time_h = time.time() - start_time
                    
                    st.metric("HGVNS Makespan", f"{sol_h['fitness']:.2f}", f"{time_h:.2f} s", delta_color="off")
                    st.plotly_chart(draw_interactive_map(parsed_data.nodes, sol_h['truck_route'], sol_h['drone_trips'], "HGVNS"), use_container_width=True)
                
                st.divider()
                st.subheader("📊 Detailed Benchmark Report")
                
                gap_val = sol_h['fitness'] - sol_b['fitness']
                gap_pct = (gap_val / sol_h['fitness']) * 100 if sol_h['fitness'] > 0 else 0
                
                c_rep1, c_rep2, c_rep3, c_rep4 = st.columns(4)
                if gap_val > 0:
                    c_rep1.success(f"🏆 **Winner: BRKGA**\n\nOutperformed HGVNS by **{gap_pct:.2f}%**")
                else:
                    c_rep1.error(f"🏆 **Winner: HGVNS**\n\nOutperformed BRKGA by **{abs(gap_pct):.2f}%**")
                    
                c_rep2.info(f"⏱️ **Computation Time**\n\nBRKGA: {time_b:.2f}s | HGVNS: {time_h:.2f}s")
                c_rep3.info(f"🚁 **Drone Utilization**\n\nBRKGA: {len(sol_b['drone_trips'])} trips | HGVNS: {len(sol_h['drone_trips'])} trips")
                
                if baseline_tsp_cost > 0:
                    h_savings = ((baseline_tsp_cost - sol_h['fitness']) / baseline_tsp_cost) * 100
                    b_savings = ((baseline_tsp_cost - sol_b['fitness']) / baseline_tsp_cost) * 100
                    c_rep4.info(f"💰 **Savings vs Baseline TSP**\n\nBRKGA: **{b_savings:.1f}%**\n\nHGVNS: **{h_savings:.1f}%**")
                
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
                sol = HGVNS_Engine(parsed_data, custom_tsp=parsed_tsp).run(pb, st_txt, hgvns_stop_type, hgvns_stop_val)
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

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
# 2. MODULE: BRKGA, PMA & HGVNS FAST EVAL (NUMBA)
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

@njit
def fast_eval_hgvns(truck_route, drone_trips, t_matrix, d_matrix, num_nodes, max_fly):
    M = len(drone_trips)
    if (len(truck_route) - 2 + M) != (num_nodes - 1):
        return np.inf

    idx_map = np.full(num_nodes, -1, dtype=np.int32)
    idx_map[0] = 0 
    
    for i in range(1, len(truck_route)):
        node = truck_route[i]
        if node != 0:
            idx_map[node] = i
            
    valid_map_count = 0
    for i in range(num_nodes):
        if idx_map[i] != -1:
            valid_map_count += 1
            
    if valid_map_count != len(truck_route) - 1:
        return np.inf

    trip_l_idx = np.zeros(M, dtype=np.int32)
    trip_r_idx = np.zeros(M, dtype=np.int32)
    trip_dt = np.zeros(M, dtype=np.float64)

    for i in range(M):
        l = drone_trips[i, 0]
        n = drone_trips[i, 1]
        r = drone_trips[i, 2]

        l_idx = 0 if l == 0 else idx_map[l]
        r_idx = len(truck_route) - 1 if r == 0 else idx_map[r]

        if l_idx == -1 or r_idx == -1 or l_idx >= r_idx:
            return np.inf
            
        dt = d_matrix[l, n] + d_matrix[n, r]
        if dt > max_fly:
            return np.inf
            
        trip_l_idx[i] = l_idx
        trip_r_idx[i] = r_idx
        trip_dt[i] = dt

    for i in range(M - 1):
        for j in range(i + 1, M):
            if trip_l_idx[i] > trip_l_idx[j]:
                trip_l_idx[i], trip_l_idx[j] = trip_l_idx[j], trip_l_idx[i]
                trip_r_idx[i], trip_r_idx[j] = trip_r_idx[j], trip_r_idx[i]
                trip_dt[i], trip_dt[j] = trip_dt[j], trip_dt[i]

    for i in range(M - 1):
        if trip_l_idx[i + 1] < trip_r_idx[i]:
            return np.inf

    cost = 0.0
    r_idx = 0
    t_idx = 0
    
    while r_idx < len(truck_route) - 1:
        if t_idx < M and trip_l_idx[t_idx] == r_idx:
            ret_idx = trip_r_idx[t_idx]
            dt = trip_dt[t_idx]
            
            tt = 0.0
            for k in range(r_idx, ret_idx):
                tt += t_matrix[truck_route[k], truck_route[k+1]]
                
            cost += tt if tt > dt else dt
            r_idx = ret_idx
            t_idx += 1
        else:
            cost += t_matrix[truck_route[r_idx], truck_route[r_idx+1]]
            r_idx += 1

    if t_idx != M:
        return np.inf

    return cost

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

    def run(self, progress_bar, status_text, use_2opt, use_3opt, use_mass_extinction=False, stagnation_limit=20, elite_survivors=3, log_console=None):
        if log_console: log_console.write("🧬 **BRKGA Engine Initialized:** Creating initial population...")
        population = [self.create_individual() for _ in range(self.p)]
        
        for ind in population: self.evaluate(ind, use_2opt, use_3opt)
        best_solution = None
        
        # Nabız Ölçer (Stagnation Counter) Değişkenleri
        stagnation_counter = 0
        last_best_fitness = float('inf')

        for gen in range(self.max_gen):
            population.sort(key=lambda x: x['fitness'])
            current_best = population[0]['fitness']
            
            if best_solution is None or current_best < best_solution['fitness']:
                if best_solution is not None and log_console:
                    log_console.write(f"🎉 **Gen {gen+1}:** New Best Makespan Found! `{best_solution['fitness']:.2f}` ➔ `{current_best:.2f}`")
                best_solution = population[0].copy()
                
            # Stagnation Control (Tıkanıklık Kontrolü)
            if current_best < last_best_fitness - 1e-4:
                stagnation_counter = 0
                last_best_fitness = current_best
            else:
                stagnation_counter += 1
                
            # Mass Extinction (Kıyamet Mutasyonu) Tetikleyicisi
            if use_mass_extinction and stagnation_counter >= stagnation_limit:
                if log_console:
                    log_console.write(f"🌋 **Gen {gen+1}: Mass Extinction Triggered!** (Stagnation for {stagnation_limit} gens). Saving top {elite_survivors} elites and resetting the rest.")
                
                survivors = population[:elite_survivors]
                new_blood = [self.create_individual() for _ in range(self.p - elite_survivors)]
                for ind in new_blood:
                    self.evaluate(ind, use_2opt, use_3opt)
                
                population = survivors + new_blood
                stagnation_counter = 0
                continue # Bu jenerasyonda çaprazlama yapma, yeni rastgeleliği değerlendir
                
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
            
            if log_console and (gen + 1) % 50 == 0:
                log_console.write(f"⏳ **Gen {gen+1} Status Update:** Current Best Makespan = `{best_solution['fitness']:.2f}`")
                
        if log_console: log_console.write("✅ **Optimization Complete!** Final best solution preserved.")
        return best_solution


# ==========================================
# 3. MODULE: HGVNS ENGINE (NUMBA ACCELERATED & DEEP INTERRUPT)
# ==========================================
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
        if len(drone_trips) > 0:
            d_arr = np.array(drone_trips, dtype=np.int32)
        else:
            d_arr = np.zeros((0, 3), dtype=np.int32)
            
        return fast_eval_hgvns(t_arr, d_arr, self.t, self.d, self.num_nodes, self.max_fly)

    def get_valid_shake(self, base_t, base_d, k_shake):
        shaken_t, shaken_d = list(base_t), list(base_d)
        for _ in range(k_shake):
            action = random.choice(['swap', 'drop'])
            if action == 'drop' and shaken_d:
                trip = random.choice(shaken_d)
                shaken_d.remove(trip)
                shaken_t.insert(random.randint(1, len(shaken_t)-1), trip[1])
            elif len(shaken_t) > 3:
                i, j = random.sample(range(1, len(shaken_t)-1), 2)
                shaken_t[i], shaken_t[j] = shaken_t[j], shaken_t[i]
        
        while self.evaluate_cost(shaken_t, shaken_d) == float('inf') and shaken_d:
            trip = shaken_d.pop(random.randrange(len(shaken_d)))
            shaken_t.insert(random.randint(1, len(shaken_t)-1), trip[1])
            
        return shaken_t, shaken_d

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
                    l = temp_t[l_idx]
                    if self.d[l][node] > self.max_fly: continue 
                    for r_idx in range(l_idx+1, min(l_idx+15, len(temp_t))):
                        r = temp_t[r_idx]
                        if self.d[l][node] + self.d[node][r] > self.max_fly: continue 
                        
                        cand_d = drone_trips[:] + [(l, node, r)]
                        c = self.evaluate_cost(temp_t, cand_d)
                        
                        if c < best_c:
                            best_c = c
                            best_move = (temp_t, cand_d)
                            
            if best_move:
                truck_route, drone_trips = best_move
                improved = True
                
        return truck_route, drone_trips

    def algorithm4_rvnd(self, truck_route, drone_trips, start_time, stop_type, stop_val):
        best_t, best_d = list(truck_route), list(drone_trips)
        best_cost = self.evaluate_cost(best_t, best_d)
        neighborhoods = [1, 2, 3, 4, 5, 6, 7]
        random.shuffle(neighborhoods)
        
        k = 0
        while k < 7:
            if stop_type == "Time Budget (sec)" and time.time() - start_time >= stop_val:
                return best_t, best_d, best_cost
                
            n_idx = neighborhoods[k]
            improved = False
            mixed_nodes = {l for l,v,r in best_d} | {r for l,v,r in best_d}
            
            if n_idx == 1: 
                for i in range(1, len(best_t)-1):
                    for j in range(1, len(best_t)):
                        if i == j or i == j-1: continue
                        new_t = best_t[:]
                        node = new_t.pop(i)
                        new_t.insert(j if j < i else j-1, node)
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost: best_cost, best_t, improved = c, new_t, True; break
                    if improved: break
                            
            elif n_idx == 2: 
                for i in range(1, len(best_t)-2):
                    for j in range(1, len(best_t)):
                        if j in [i, i+1, i+2]: continue
                        new_t = best_t[:]
                        n1 = new_t.pop(i)
                        n2 = new_t.pop(i)
                        ins = j if j < i else j-2
                        new_t.insert(ins, n2)
                        new_t.insert(ins, n1)
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost: best_cost, best_t, improved = c, new_t, True; break
                    if improved: break

            elif n_idx == 3: 
                for i in range(1, len(best_t)-1):
                    for j in range(i+1, len(best_t)-1):
                        new_t = best_t[:]
                        new_t[i], new_t[j] = new_t[j], new_t[i]
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost: best_cost, best_t, improved = c, new_t, True; break
                    if improved: break
                if improved: k=0; continue

                if len(best_d) > 0:
                    for i in range(1, len(best_t)-1):
                        if best_t[i] in mixed_nodes: continue 
                        for j in range(len(best_d)):
                            new_t, new_d = best_t[:], best_d[:]
                            t_node = new_t[i]
                            l, d_node, r = new_d[j]
                            if self.d[l][t_node] + self.d[t_node][r] > self.max_fly: continue
                            
                            new_t[i] = d_node
                            new_d[j] = (l, t_node, r)
                            c = self.evaluate_cost(new_t, new_d)
                            if c < best_cost: best_cost, best_t, best_d, improved = c, new_t, new_d, True; break
                        if improved: break
                if improved: k=0; continue

                if len(best_d) > 1:
                    for i in range(len(best_d)-1):
                        for j in range(i+1, len(best_d)):
                            new_d = best_d[:]
                            l1, n1, r1 = new_d[i]
                            l2, n2, r2 = new_d[j]
                            if self.d[l1][n2] + self.d[n2][r1] > self.max_fly: continue
                            if self.d[l2][n1] + self.d[n1][r2] > self.max_fly: continue
                            
                            new_d[i] = (l1, n2, r1)
                            new_d[j] = (l2, n1, r2)
                            c = self.evaluate_cost(best_t, new_d)
                            if c < best_cost: best_cost, best_d, improved = c, new_d, True; break
                        if improved: break

            elif n_idx == 4: 
                for i in range(1, len(best_t)-2):
                    if stop_type == "Time Budget (sec)" and time.time() - start_time >= stop_val:
                        return best_t, best_d, best_cost
                    for j in range(1, len(best_t)-1):
                        if j == i or j == i+1: continue
                        new_t = best_t[:]
                        if i < j:
                            nj = new_t.pop(j)
                            ni1 = new_t.pop(i)
                            ni2 = new_t.pop(i)
                            new_t.insert(i, nj)
                            new_t.insert(j-1, ni2)
                            new_t.insert(j-1, ni1)
                        else:
                            ni1 = new_t.pop(i)
                            ni2 = new_t.pop(i)
                            nj = new_t.pop(j)
                            new_t.insert(j, ni2)
                            new_t.insert(j, ni1)
                            new_t.insert(i+1, nj)
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost: best_cost, best_t, improved = c, new_t, True; break
                    if improved: break

            elif n_idx == 5: 
                for i in range(1, len(best_t)-2):
                    if stop_type == "Time Budget (sec)" and time.time() - start_time >= stop_val:
                        return best_t, best_d, best_cost
                    for j in range(i+2, len(best_t)-2):
                        new_t = best_t[:]
                        new_t[i:i+2], new_t[j:j+2] = new_t[j:j+2], new_t[i:i+2]
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost: best_cost, best_t, improved = c, new_t, True; break
                    if improved: break

            elif n_idx == 6: 
                for i in range(1, len(best_t)-2):
                    for j in range(i+1, len(best_t)-1):
                        n1, n2 = best_t[i-1], best_t[i]
                        n3, n4 = best_t[j], best_t[j+1]
                        
                        if self.t[n1][n3] + self.t[n2][n4] >= self.t[n1][n2] + self.t[n3][n4]:
                            continue
                            
                        new_t = best_t[:i] + best_t[i:j+1][::-1] + best_t[j+1:]
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost: best_cost, best_t, improved = c, new_t, True; break
                    if improved: break

            elif n_idx == 7: 
                for i in range(1, len(best_t)-1):
                    if stop_type == "Time Budget (sec)" and time.time() - start_time >= stop_val:
                        return best_t, best_d, best_cost
                        
                    node = best_t[i]
                    if node in self.novisit_list or node in mixed_nodes: continue 
                    
                    temp_t = best_t[:i] + best_t[i+1:]
                    best_insert_c = float('inf')
                    best_insert_d, best_insert_t = best_d, best_t
                    
                    for l_idx in range(len(temp_t)-1):
                        l = temp_t[l_idx]
                        if self.d[l][node] > self.max_fly: continue 
                        for r_idx in range(l_idx+1, min(l_idx+15, len(temp_t))):
                            r = temp_t[r_idx]
                            if self.d[l][node] + self.d[node][r] > self.max_fly: continue 
                            
                            cand_d = best_d[:] + [(l, node, r)]
                            c = self.evaluate_cost(temp_t, cand_d)
                            if c < best_insert_c:
                                best_insert_c, best_insert_t, best_insert_d = c, temp_t, cand_d
                    
                    if best_insert_c < best_cost:
                        best_cost, best_t, best_d, improved = best_insert_c, best_insert_t, best_insert_d, True
                        break
                if improved: k=0; continue

                if len(best_d) > 0:
                    for i, trip in enumerate(best_d):
                        if stop_type == "Time Budget (sec)" and time.time() - start_time >= stop_val:
                            return best_t, best_d, best_cost
                            
                        temp_d = best_d[:i] + best_d[i+1:]
                        node = trip[1]
                        best_insert_c = float('inf')
                        best_insert_t, best_insert_d = best_t, best_d
                        
                        for j in range(1, len(best_t)):
                            new_t = best_t[:]
                            new_t.insert(j, node)
                            c = self.evaluate_cost(new_t, temp_d)
                            if c < best_insert_c:
                                best_insert_c, best_insert_t, best_insert_d = c, new_t, temp_d
                                
                        if best_insert_c < best_cost:
                            best_cost, best_t, best_d, improved = best_insert_c, best_insert_t, best_insert_d, True
                            break
                if improved: k=0; continue
                
                if len(best_d) > 0:
                    for i, trip in enumerate(best_d):
                        node = trip[1]
                        temp_d = best_d[:i] + best_d[i+1:]
                        best_insert_c = float('inf')
                        best_insert_d = best_d
                        
                        for l_idx in range(len(best_t)-1):
                            l = best_t[l_idx]
                            if self.d[l][node] > self.max_fly: continue
                            for r_idx in range(l_idx+1, min(l_idx+15, len(best_t))):
                                r = best_t[r_idx]
                                if self.d[l][node] + self.d[node][r] > self.max_fly: continue
                                
                                cand_d = temp_d[:] + [(l, node, r)]
                                c = self.evaluate_cost(best_t, cand_d)
                                if c < best_insert_c:
                                    best_insert_c, best_insert_d = c, cand_d
                        
                        if best_insert_c < best_cost:
                            best_cost, best_d, improved = best_insert_c, best_insert_d, True
                            break

            if improved:
                random.shuffle(neighborhoods)
                k = 0
            else:
                k += 1
                
        return best_t, best_d, best_cost

    def run(self, progress_bar, status_text, stop_type, stop_val):
        if status_text: status_text.text("HGVNS: Alg 1 & 2 (Concorde Initial & Global Savings)...")
        best_t, best_d = self.algorithm2_initial_solution()
        best_cost = self.evaluate_cost(best_t, best_d)
        
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
                
            if progress_bar:
                if stop_type == "Max Iterations": progress_bar.progress(min(iter_count / stop_val, 1.0))
                elif stop_type == "Time Budget (sec)": progress_bar.progress(min(elapsed / stop_val, 1.0))
                else: progress_bar.progress(min(no_improve_count / stop_val, 1.0))
            
            if status_text and iter_count % 10 == 0: 
                status_text.text(f"HGVNS Search... Iter: {iter_count} | No Improve: {no_improve_count} | Best: {best_cost:.2f}")

            shaken_t, shaken_d = self.get_valid_shake(best_t, best_d, k_shake)
            
            new_t, new_d, new_cost = self.algorithm4_rvnd(shaken_t, shaken_d, start_time, stop_type, stop_val)
            
            if new_cost < best_cost - 1e-4:
                best_cost = new_cost
                best_t, best_d = new_t, new_d
                k_shake = 1
                no_improve_count = 0
            else:
                k_shake += 1
                no_improve_count += 1
                if k_shake > k_max: k_shake = 1
                    
            iter_count += 1
            
        if progress_bar: progress_bar.progress(1.0)
        if status_text: status_text.text(f"HGVNS Completed! Makespan: {best_cost:.2f}")
        
        return {'fitness': best_cost, 'truck_route': best_t, 'drone_trips': best_d}


# ==========================================
# 4. HELPER: WAIT TIME CALCULATOR
# ==========================================
def calculate_wait_times(truck_route, drone_trips, t_matrix, d_matrix):
    truck_wait = 0.0
    drone_wait = 0.0
    
    idx_map = {node: i for i, node in enumerate(truck_route)}
    
    for l, n, r in drone_trips:
        dt = d_matrix[l][n] + d_matrix[n][r]
        
        l_idx = idx_map.get(l, -1)
        r_idx = idx_map.get(r, -1)
        
        if l_idx != -1 and r_idx != -1 and l_idx < r_idx:
            tt = 0.0
            for k in range(l_idx, r_idx):
                tt += t_matrix[truck_route[k]][truck_route[k+1]]
                
            if dt > tt:
                truck_wait += (dt - tt)
            else:
                drone_wait += (tt - dt)
                
    return truck_wait, drone_wait


# ==========================================
# 5. INTERACTIVE MAP (PLOTLY)
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

st.sidebar.subheader("Local Search Heuristics (BRKGA)")
use_2opt = st.sidebar.checkbox("Enable 2-Opt Local Search", value=False)
use_3opt = st.sidebar.checkbox("Enable 3-Opt Local Search", value=False)
st.sidebar.caption("⚠️ Pro Tip: 3-Opt is computationally heavy. Safe mode (PMA) is active.")

# YENİ EKLENTİ: MASS EXTINCTION ARAYÜZÜ
st.sidebar.divider()
st.sidebar.subheader("🧬 Advanced Evolutionary Features")
use_mass_extinction = st.sidebar.checkbox("Enable Mass Extinction (Partial Reinitialization)", value=False)
stagnation_limit = st.sidebar.number_input("Stagnation Limit (Generations)", min_value=5, max_value=100, value=20, step=5)
elite_survivors = st.sidebar.number_input("Noah's Ark (Elites to save)", min_value=1, max_value=10, value=3, step=1)

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
                    sol_b = BRKGA_Engine(pop_size, elite_ratio, mutant_ratio, rho_e, max_gen, DPSplitDecoder(parsed_data)).run(
                        pb_b, st_txt_b, use_2opt, use_3opt, use_mass_extinction, stagnation_limit, elite_survivors, log_b
                    )
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
                
                cw1, cw2 = st.columns(2)
                t_wait_b, d_wait_b = calculate_wait_times(sol_b['truck_route'], sol_b['drone_trips'], parsed_data.truck_time_matrix, parsed_data.drone_time_matrix)
                t_wait_h, d_wait_h = calculate_wait_times(sol_h['truck_route'], sol_h['drone_trips'], parsed_data.truck_time_matrix, parsed_data.drone_time_matrix)
                
                cw1.info(f"⏳ **BRKGA Wait Times (Idle)**\n\n🚚 Truck Waited: **{t_wait_b:.2f}**\n\n🚁 Drone Waited: **{d_wait_b:.2f}**")
                cw2.info(f"⏳ **HGVNS Wait Times (Idle)**\n\n🚚 Truck Waited: **{t_wait_h:.2f}**\n\n🚁 Drone Waited: **{d_wait_h:.2f}**")

                st.write("---")
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown("#### 🟦 BRKGA Routes")
                    st.caption("**Truck Route:** " + " ➔ ".join(map(str, sol_b['truck_route'])))
                    
                    truck_idx_b = {}
                    for idx, node in enumerate(sol_b['truck_route']):
                        if node not in truck_idx_b:
                            truck_idx_b[node] = idx
                    sorted_d_trips_b = sorted(sol_b['drone_trips'], key=lambda x: truck_idx_b.get(x[0], 999))
                    
                    for i, t_trip in enumerate(sorted_d_trips_b):
                        st.caption(f"**Drone Trip {i+1}:** Node {t_trip[0]} ➔ **Visit {t_trip[1]}** ➔ Node {t_trip[2]}")
                        
                with col_r2:
                    st.markdown("#### 🟥 HGVNS Routes")
                    st.caption("**Truck Route:** " + " ➔ ".join(map(str, sol_h['truck_route'])))
                    
                    truck_idx_h = {}
                    for idx, node in enumerate(sol_h['truck_route']):
                        if node not in truck_idx_h:
                            truck_idx_h[node] = idx
                    sorted_d_trips_h = sorted(sol_h['drone_trips'], key=lambda x: truck_idx_h.get(x[0], 999))
                    
                    for i, t_trip in enumerate(sorted_d_trips_h):
                        st.caption(f"**Drone Trip {i+1}:** Node {t_trip[0]} ➔ **Visit {t_trip[1]}** ➔ Node {t_trip[2]}")

            elif solver_type == "BRKGA (Memetic)":
                st.subheader("🟦 BRKGA Engine Results")
                pb, st_txt = st.progress(0), st.empty()
                
                metric_placeholder = st.empty()
                map_placeholder = st.empty()
                log_b = st.expander("🔍 Inside the BRKGA Brain (Live Process Logs)", expanded=False)
                
                start_time = time.time()
                sol = BRKGA_Engine(pop_size, elite_ratio, mutant_ratio, rho_e, max_gen, DPSplitDecoder(parsed_data)).run(
                    pb, st_txt, use_2opt, use_3opt, use_mass_extinction, stagnation_limit, elite_survivors, log_b
                )
                elapsed = time.time() - start_time
                
                t_w, d_w = calculate_wait_times(sol['truck_route'], sol['drone_trips'], parsed_data.truck_time_matrix, parsed_data.drone_time_matrix)
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Makespan", f"{sol['fitness']:.2f}")
                c2.metric("Time", f"{elapsed:.2f} s")
                c3.metric("Trips", len(sol['drone_trips']))
                c4.metric("🚚 Wait Time", f"{t_w:.2f}")
                c5.metric("🚁 Wait Time", f"{d_w:.2f}")

                map_placeholder.plotly_chart(draw_interactive_map(parsed_data.nodes, sol['truck_route'], sol['drone_trips'], "BRKGA"), use_container_width=True)
                
                st.markdown("#### 📝 Route Details")
                st.write("**Truck Route:** " + " ➔ ".join(map(str, sol['truck_route'])))
                
                truck_idx = {}
                for idx, node in enumerate(sol['truck_route']):
                    if node not in truck_idx:
                        truck_idx[node] = idx
                sorted_d_trips = sorted(sol['drone_trips'], key=lambda x: truck_idx.get(x[0], 999))
                
                for i, t_trip in enumerate(sorted_d_trips):
                    st.write(f"**Drone Trip {i+1}:** Node {t_trip[0]} ➔ **Visit {t_trip[1]}** ➔ Node {t_trip[2]}")

            elif solver_type == "HGVNS (Paper Replica)":
                st.subheader("🟥 HGVNS Engine Results")
                pb, st_txt = st.progress(0), st.empty()
                
                start_time = time.time()
                sol = HGVNS_Engine(parsed_data, custom_tsp=parsed_tsp).run(pb, st_txt, hgvns_stop_type, hgvns_stop_val)
                elapsed = time.time() - start_time
                
                t_w, d_w = calculate_wait_times(sol['truck_route'], sol['drone_trips'], parsed_data.truck_time_matrix, parsed_data.drone_time_matrix)
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Makespan", f"{sol['fitness']:.2f}")
                c2.metric("Time", f"{elapsed:.2f} s")
                c3.metric("Trips", len(sol['drone_trips']))
                c4.metric("🚚 Wait Time", f"{t_w:.2f}")
                c5.metric("🚁 Wait Time", f"{d_w:.2f}")

                st.plotly_chart(draw_interactive_map(parsed_data.nodes, sol['truck_route'], sol['drone_trips'], "HGVNS"), use_container_width=True)
                
                st.markdown("#### 📝 Route Details")
                st.write("**Truck Route:** " + " ➔ ".join(map(str, sol['truck_route'])))
                
                truck_idx = {}
                for idx, node in enumerate(sol['truck_route']):
                    if node not in truck_idx:
                        truck_idx[node] = idx
                sorted_d_trips = sorted(sol['drone_trips'], key=lambda x: truck_idx.get(x[0], 999))
                
                for i, t_trip in enumerate(sorted_d_trips):
                    st.write(f"**Drone Trip {i+1}:** Node {t_trip[0]} ➔ **Visit {t_trip[1]}** ➔ Node {t_trip[2]}")

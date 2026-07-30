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
        for trip in drone_trips:
            if trip[0] not in truck_route or trip[2] not in truck_route:
                return float('inf')
            if truck_route.index(trip[0]) >= truck_route.index(trip[2]):
                return float('inf')
        try:
            sorted_trips = sorted(drone_trips, key=lambda x: truck_route.index(x[0]))
        except ValueError:
            return float('inf')

        for i in range(len(sorted_trips)-1):
            t1, t2 = sorted_trips[i], sorted_trips[i+1]
            idx_t1_ret = truck_route.index(t1[2])
            idx_t2_launch = truck_route.index(t2[0])
            if idx_t2_launch < idx_t1_ret: return float('inf')

        total_cost = 0.0
        curr_trip_idx = 0
        i = 0
        
        while i < len(truck_route) - 1:
            u = truck_route[i]
            
            if curr_trip_idx < len(sorted_trips) and sorted_trips[curr_trip_idx][0] == u:
                trip = sorted_trips[curr_trip_idx]
                drone_node = trip[1]
                return_node = trip[2]
                
                try:
                    ret_idx = truck_route.index(return_node, i + 1)
                except ValueError:
                    return float('inf')
                    
                truck_time = 0.0
                for k in range(i, ret_idx):
                    truck_time += self.t[truck_route[k]][truck_route[k+1]]
                    
                drone_time = self.d[u][drone_node] + self.d[drone_node][return_node]
                if drone_time > self.max_fly: 
                    return float('inf')
                    
                total_cost += max(truck_time, drone_time)
                i = ret_idx
                curr_trip_idx += 1
            else:
                total_cost += self.t[truck_route[i]][truck_route[i+1]]
                i += 1
                
        return total_cost

    def _solve_tsp(self):
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
                temp_d = drone_trips + [(prev_n, node, next_n)]
                
                # Gerçek Global Maliyet Hesaplaması (Daha güvenli!)
                c = self.evaluate_cost(temp_t, temp_d)
                
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
                
        drone_trips.sort(key=lambda x: truck_route.index(x[0]))
        return truck_route, drone_trips

    def algorithm4_rvnd(self, truck_route, drone_trips):
        best_t, best_d = truck_route.copy(), drone_trips.copy()
        best_cost = self.evaluate_cost(best_t, best_d)
        
        # Makaledeki 5 kritik komşuluk (Özellikle 5 numara hayat kurtaracak)
        neighborhoods = [1, 2, 3, 4, 5] 
        random.shuffle(neighborhoods)
        
        k = 0
        while k < len(neighborhoods):
            n_idx = neighborhoods[k]
            improved = False
            
            if n_idx == 1: # 2-opt
                for i in range(1, len(best_t)-2):
                    for j in range(i+1, len(best_t)-1):
                        new_t = best_t[:i] + best_t[i:j+1][::-1] + best_t[j+1:]
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost:
                            best_cost, best_t = c, new_t
                            improved = True
                            break
                    if improved: break
                            
            elif n_idx == 2: # Swap
                for i in range(1, len(best_t)-1):
                    for j in range(i+1, len(best_t)-1):
                        new_t = best_t.copy()
                        new_t[i], new_t[j] = new_t[j], new_t[i]
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost:
                            best_cost, best_t = c, new_t
                            improved = True
                            break
                    if improved: break

            elif n_idx == 3: # Reinsert
                for i in range(1, len(best_t)-1):
                    node = best_t[i]
                    temp_t = best_t[:i] + best_t[i+1:]
                    for j in range(1, len(temp_t)):
                        if i == j: continue
                        new_t = temp_t[:j] + [node] + temp_t[j:]
                        c = self.evaluate_cost(new_t, best_d)
                        if c < best_cost:
                            best_cost, best_t = c, new_t
                            improved = True
                            break
                    if improved: break
            
            elif n_idx == 4 and len(best_d) > 0: # Drone'dan Kamyona İade
                for i, trip in enumerate(best_d):
                    new_d = best_d[:i] + best_d[i+1:]
                    node = trip[1]
                    best_insert_c = float('inf')
                    best_insert_t = best_t
                    for j in range(1, len(best_t)):
                        new_t = best_t[:j] + [node] + best_t[j:]
                        c = self.evaluate_cost(new_t, new_d)
                        if c < best_insert_c:
                            best_insert_c, best_insert_t = c, new_t
                    
                    if best_insert_c < best_cost:
                        best_cost, best_t, best_d = best_insert_c, best_insert_t, new_d
                        improved = True
                        break

            elif n_idx == 5: # YENİ SİLAH: Kamyon'dan Drone'a Gönderim (Paper 4.8)
                for i in range(1, len(best_t)-1):
                    drone_cand = best_t[i]
                    if drone_cand in self.novisit_list: continue
                    
                    temp_t = best_t[:i] + best_t[i+1:]
                    best_insert_c = float('inf')
                    best_insert_d = best_d
                    
                    for launch_idx in range(len(temp_t) - 1):
                        for ret_idx in range(launch_idx + 1, len(temp_t)):
                            launch_node = temp_t[launch_idx]
                            ret_node = temp_t[ret_idx]
                            
                            d_time = self.d[launch_node][drone_cand] + self.d[drone_cand][ret_node]
                            if d_time > self.max_fly: continue
                            
                            temp_d = best_d.copy()
                            temp_d.append((launch_node, drone_cand, ret_node))
                            
                            c = self.evaluate_cost(temp_t, temp_d)
                            if c < best_insert_c:
                                best_insert_c = c
                                best_insert_d = temp_d
                                
                    if best_insert_c < best_cost:
                        best_cost = best_insert_c
                        best_t = temp_t
                        best_d = best_insert_d
                        improved = True
                        break

            if improved:
                random.shuffle(neighborhoods)
                k = 0
            else:
                k += 1
                
        return best_t, best_d, best_cost

    def run(self, progress_bar, status_text):
        if status_text: status_text.text("HGVNS: Alg 1 & 2 (Concorde-like Initial & Global Savings)...")
        best_t, best_d = self.algorithm2_initial_solution()
        best_cost = self.evaluate_cost(best_t, best_d)
        
        max_iters = 100  # Makine ısınsın diye 30'dan 100'e çıkardım
        k_max = 5
        k_shake = 1
        
        for iter_count in range(max_iters):
            if progress_bar: progress_bar.progress((iter_count + 1) / max_iters)
            if status_text: status_text.text(f"HGVNS: Alg 3 & 4 (GVNS Search)... Iter: {iter_count+1}/{max_iters} | Score: {best_cost:.2f}")
            
            shaken_t, shaken_d = best_t.copy(), best_d.copy()
            for _ in range(k_shake):
                if len(shaken_t) > 3:
                    idx1, idx2 = random.sample(range(1, len(shaken_t)-1), 2)
                    shaken_t[idx1], shaken_t[idx2] = shaken_t[idx2], shaken_t[idx1]
            
            new_t, new_d, new_cost = self.algorithm4_rvnd(shaken_t, shaken_d)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_t, best_d = new_t, new_d
                k_shake = 1
            else:
                k_shake += 1
                if k_shake > k_max:
                    k_shake = 1
        
        if status_text: status_text.text(f"HGVNS Completed! Makespan: {best_cost:.2f}")
        return {'fitness': best_cost, 'truck_route': best_t, 'drone_trips': best_d}

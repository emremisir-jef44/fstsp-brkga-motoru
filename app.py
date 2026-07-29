import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import math
import random
import time
import os  # BU YENİ EKLENDİ

# ==========================================
# 1. MODÜL: DİNAMİK VERİ OKUYUCU (PARSER)
# ==========================================
class FSTSP_Parser:
    def __init__(self, file_content):
        self.max_fly = float('inf') 
        self.novisit_list = []
        self.truck_speed = 1.0 # Aslında Faktör (Çarpan)
        self.drone_speed = 1.0 # Aslında Faktör (Çarpan)
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
                    
                    # DÜZELTME 1: Hızlar faktör olduğu için BÖLME değil ÇARPMA yapılır. (0.5 ise süre yarıya iner)
                    t_matrix[i][j] = dist * self.truck_speed
                    d_matrix[i][j] = dist * self.drone_speed
        return t_matrix, d_matrix

# ==========================================
# 2. MODÜL: AKILLI ÇÖZÜCÜ (SMART DECODER - DAG)
# ==========================================
class SmartDecoder:
    def __init__(self, parsed_data):
        self.data = parsed_data
        self.t_matrix = parsed_data.truck_time_matrix
        self.d_matrix = parsed_data.drone_time_matrix

    def decode(self, rk_route, rk_modes):
        customers = list(range(1, self.data.num_nodes))
        sorted_customers = [cust for _, cust in sorted(zip(rk_route, customers))]
        
        modes = {0: 'T'}
        for i, cust in enumerate(customers):
            if cust in self.data.novisit_list:
                modes[cust] = 'T'
            else:
                modes[cust] = 'D' if rk_modes[i] >= 0.5 else 'T'
                
        # DÜZELTME 2: REPAIR (Onarım). Peş peşe gelen D'leri engelle ki yollar kopmasın (inf hatası çözümü)
        last_mode = 'T'
        for cust in sorted_customers:
            if modes[cust] == 'D':
                if last_mode == 'D':
                    modes[cust] = 'T' # Çakışmayı önle
                    last_mode = 'T'
                else:
                    last_mode = 'D'
            else:
                last_mode = 'T'
                
        sequence = [0] + sorted_customers + [0]
        t_nodes_in_seq = [idx for idx, node in enumerate(sequence) if modes[node] == 'T']
        
        G = nx.DiGraph()
        G.add_nodes_from(t_nodes_in_seq)
        
        for i in range(len(t_nodes_in_seq) - 1):
            idx_start = t_nodes_in_seq[i]
            for j in range(i + 1, len(t_nodes_in_seq)):
                idx_end = t_nodes_in_seq[j]
                node_u, node_v = sequence[idx_start], sequence[idx_end]
                
                sub_seq = sequence[idx_start+1 : idx_end]
                d_nodes = [n for n in sub_seq if modes[n] == 'D']
                t_nodes = [n for n in sub_seq if modes[n] == 'T']
                
                if len(d_nodes) == 0 and j == i + 1:
                    cost = self.t_matrix[node_u][node_v]
                    G.add_edge(idx_start, idx_end, weight=cost, drone_node=None)
                
                elif len(d_nodes) == 1 and len(t_nodes) == (j - i - 1):
                    drone_cust = d_nodes[0]
                    drone_time = self.d_matrix[node_u][drone_cust] + self.d_matrix[drone_cust][node_v]
                    
                    truck_time, curr = 0, node_u
                    for internal_node in t_nodes:
                        truck_time += self.t_matrix[curr][internal_node]
                        curr = internal_node
                    truck_time += self.t_matrix[curr][node_v]
                    
                    max_time = max(truck_time, drone_time)
                    if max_time <= self.data.max_fly:
                        G.add_edge(idx_start, idx_end, weight=max_time, drone_node=drone_cust)

        try:
            path = nx.shortest_path(G, source=0, target=len(sequence)-1, weight='weight')
            cost = nx.shortest_path_length(G, source=0, target=len(sequence)-1, weight='weight')
            
            truck_route, drone_trips = [], []
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                edge = G.get_edge_data(u, v)
                truck_route.append(sequence[u])
                if edge['drone_node'] is not None:
                    drone_trips.append((sequence[u], edge['drone_node'], sequence[v]))
            truck_route.append(sequence[path[-1]])
            
            return cost, truck_route, drone_trips
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf'), [], []

# ==========================================
# 3. MODÜL: BRKGA EVRİM MOTORU
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
            'mode': [random.random() for _ in range(self.num_cust)],
            'fitness': float('inf'), 'truck_route': [], 'drone_trips': []
        }

    def evaluate(self, ind):
        cost, t_route, d_trips = self.decoder.decode(ind['route'], ind['mode'])
        ind['fitness'] = cost
        ind['truck_route'] = t_route
        ind['drone_trips'] = d_trips

    def run(self, progress_bar, status_text):
        population = [self.create_individual() for _ in range(self.p)]
        for ind in population: self.evaluate(ind)
        
        best_solution = None

        for gen in range(self.max_gen):
            population.sort(key=lambda x: x['fitness'])
            if best_solution is None or population[0]['fitness'] < best_solution['fitness']:
                best_solution = population[0].copy()
                
            next_gen = []
            elites = population[:self.p_e]
            non_elites = population[self.p_e:]
            next_gen.extend(elites)
            
            mutants = [self.create_individual() for _ in range(self.p_m)]
            for mut in mutants: self.evaluate(mut)
            next_gen.extend(mutants)
            
            num_offspring = self.p - self.p_e - self.p_m
            for _ in range(num_offspring):
                parent_a = random.choice(elites)
                parent_b = random.choice(non_elites) if non_elites else random.choice(elites)
                
                child = {'route': [], 'mode': [], 'fitness': float('inf')}
                for i in range(self.num_cust):
                    child['route'].append(parent_a['route'][i] if random.random() < self.rho_e else parent_b['route'][i])
                    child['mode'].append(parent_a['mode'][i] if random.random() < self.rho_e else parent_b['mode'][i])
                
                self.evaluate(child)
                next_gen.append(child)
                
            population = next_gen
            
            if gen % 10 == 0:
                progress_bar.progress((gen + 1) / self.max_gen)
                status_text.text(f"Evrimleşiyor... Jenerasyon {gen+1}/{self.max_gen} | En İyi Süre: {best_solution['fitness']:.2f}")

        progress_bar.progress(1.0)
        status_text.text(f"Tamamlandı! Bulunan Optimum Süre: {best_solution['fitness']:.2f}")
        return best_solution

# ==========================================
# 4. MODÜL: İNTERAKTİF HARİTA (PLOTLY)
# ==========================================
def draw_interactive_map(nodes_data, truck_route, drone_trips):
    fig = go.Figure()
    nodes_dict = {node[0]: (node[1], node[2]) for node in nodes_data}
    
    truck_x = [nodes_dict[n][0] for n in truck_route]
    truck_y = [nodes_dict[n][1] for n in truck_route]
    fig.add_trace(go.Scatter(x=truck_x, y=truck_y, mode='lines+markers+text', name='Kamyon Rotası',
                             text=[str(n) for n in truck_route], textposition="bottom center",
                             line=dict(color='#1f77b4', width=3), marker=dict(size=10, color='#1f77b4')))
    
    for i, (launch, visit, ret) in enumerate(drone_trips):
        dx = [nodes_dict[launch][0], nodes_dict[visit][0], nodes_dict[ret][0]]
        dy = [nodes_dict[launch][1], nodes_dict[visit][1], nodes_dict[ret][1]]
        fig.add_trace(go.Scatter(x=dx, y=dy, mode='lines+markers+text', name=f'Dron (Tur {i+1})',
                                 text=["", str(visit), ""], textposition="top center",
                                 line=dict(color='#d62728', width=2, dash='dashdot'), marker=dict(size=8, symbol='diamond')))
        
    fig.add_trace(go.Scatter(x=[nodes_dict[0][0]], y=[nodes_dict[0][1]], mode='markers+text', name='Depo',
                             text=["DEPO"], textposition="top center", marker=dict(size=16, color='black', symbol='square')))
    
    fig.update_layout(title="🚁 FSTSP Optimum Rota Haritası", xaxis_title="X", yaxis_title="Y", hovermode="closest",
                      plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))
    return fig

# ==========================================
# STREAMLIT WEB APP ARAYÜZÜ
# ==========================================
st.set_page_config(page_title="FSTSP BRKGA Motoru", layout="wide")
st.title("🚁 FSTSP: Dinamik BRKGA Optimizasyon Motoru")

st.sidebar.header("BRKGA Parametreleri")
pop_size = st.sidebar.slider("Popülasyon (p)", 50, 500, 100, 10)
elite_ratio = st.sidebar.slider("Elit Oranı (p_e %)", 5, 40, 20, 5)
mutant_ratio = st.sidebar.slider("Mutant Oranı (p_m %)", 5, 40, 15, 5)
rho_e = st.sidebar.slider("Yanlı Çaprazlama (ρ_e)", 0.50, 0.95, 0.70, 0.05)
max_gen = st.sidebar.number_input("Maksimum Jenerasyon", value=200, min_value=10, max_value=2000)

# --- DİNAMİK DOSYA SEÇİCİ (YENİ EKLENEN KISIM) ---
st.subheader("1. Veri Seti Seçimi")

# datasets klasöründeki tüm txt dosyalarını bul
dataset_folder = "datasets"
if os.path.exists(dataset_folder):
    available_files = [f for f in os.listdir(dataset_folder) if f.endswith('.txt')]
else:
    available_files = []

if not available_files:
    st.error(f"⚠️ '{dataset_folder}' klasörü bulunamadı veya içinde .txt dosyası yok! Lütfen GitHub'a dosyaları yükleyin.")
else:
    # Açılır menü (Selectbox) ile dosya seçimi
    selected_file = st.selectbox("Çalıştırmak istediğiniz veri setini seçin:", available_files)
    
    # Seçilen dosyanın yolunu oluştur ve oku
    file_path = os.path.join(dataset_folder, selected_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    parsed_data = FSTSP_Parser(file_content)
    st.success(f"✅ {selected_file} başarıyla yüklendi!")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Düğüm", parsed_data.num_nodes)
    col2.metric("Kamyon Çarpanı", parsed_data.truck_speed)
    col3.metric("Dron Çarpanı", parsed_data.drone_speed)
    col4.metric("Dron Batarya (MAXFLY)", "Limitsiz" if parsed_data.max_fly == float('inf') else parsed_data.max_fly)
    
    if st.button("🚀 Akıllı Çözücü ile BRKGA'yı Başlat"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        decoder = SmartDecoder(parsed_data)
        engine = BRKGA_Engine(pop_size, elite_ratio, mutant_ratio, rho_e, max_gen, decoder)
        
        start_time = time.time()
        best_sol = engine.run(progress_bar, status_text)
        elapsed_time = time.time() - start_time
        
        st.subheader("📊 Optimizasyon Sonuçları")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Optimum Süre", f"{best_sol['fitness']:.2f}")
        c2.metric("Hesaplama Süresi", f"{elapsed_time:.2f} sn")
        c3.metric("Kamyon Ziyareti", f"{len(best_sol['truck_route'])-2} Düğüm")
        c4.metric("Dron Ziyareti", f"{len(best_sol['drone_trips'])} Tur")
        
        st.plotly_chart(draw_interactive_map(parsed_data.nodes, best_sol['truck_route'], best_sol['drone_trips']), use_container_width=True)
        
        with st.expander("Detaylı Rota Dökümü"):
            st.write("**Kamyon Rotası:**", " ➔ ".join(map(str, best_sol['truck_route'])))
            for i, trip in enumerate(best_sol['drone_trips']):
                st.write(f"**Dron Turu {i+1}:** Kalkış: {trip[0]} ➔ Ziyaret: {trip[1]} ➔ Varış: {trip[2]}")

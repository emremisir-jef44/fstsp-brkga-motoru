import streamlit as st
import numpy as np
import math

# ==========================================
# 1. MODÜL: DİNAMİK VERİ OKUYUCU (PARSER)
# ==========================================
class FSTSP_Parser:
    def __init__(self, file_content):
        # Başlangıçta varsayılan/boş değerler (Dinamik olarak dolacak)
        self.max_fly = float('inf') 
        self.novisit_list = []
        self.truck_speed = 1.0
        self.drone_speed = 1.0
        self.num_nodes = 0
        self.nodes = [] # (id, x, y)
        
        self._parse(file_content)
        self.truck_time_matrix, self.drone_time_matrix = self._build_matrices()

    def _parse(self, content):
        lines = content.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # --- DİNAMİK KISITLARI OKUMA ---
            if line.startswith("#MAXFLY"):
                val = line.split()[1]
                self.max_fly = float('inf') if val.lower() == 'infinity' else float(val)
            elif line.startswith("#NOVISIT"):
                self.novisit_list.append(int(line.split()[1]))
                
            # --- DİNAMİK HIZ VE DÜĞÜM SAYILARINI OKUMA ---
            elif "The speed of the Truck" in line:
                i += 1
                self.truck_speed = float(lines[i].strip())
            elif "The speed of the Drone" in line:
                i += 1
                self.drone_speed = float(lines[i].strip())
            elif "Number of Nodes" in line:
                i += 1
                self.num_nodes = int(lines[i].strip())
                
            # --- DİNAMİK KOORDİNATLARI OKUMA ---
            elif "The Depot" in line:
                i += 1
                parts = lines[i].strip().split()
                # Depo her zaman ID: 0 olarak kaydedilir
                self.nodes.append((0, float(parts[0]), float(parts[1]))) 
            elif "The Locations" in line:
                i += 1
                # Kalan satırlar müşteridir (1'den num_nodes'a kadar)
                for j in range(1, self.num_nodes):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        self.nodes.append((j, float(parts[0]), float(parts[1])))
                        i += 1
                break # Dosya sonu
            i += 1

    def _build_matrices(self):
        # Okunan hızlara göre dinamik süre matrisi oluşturulur
        n = len(self.nodes)
        t_matrix = np.zeros((n, n))
        d_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.nodes[i][1], self.nodes[i][2]
                    x2, y2 = self.nodes[j][1], self.nodes[j][2]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    t_matrix[i][j] = distance / self.truck_speed
                    d_matrix[i][j] = distance / self.drone_speed
                    
        return t_matrix, d_matrix

# ==========================================
# STREAMLIT WEB APP ARAYÜZÜ
# ==========================================
st.set_page_config(page_title="FSTSP BRKGA Motoru", layout="wide")
st.title("🚁 FSTSP: Dinamik BRKGA Optimizasyon Motoru")
st.markdown("Bu sistem tüm hızları, koordinatları ve drone kısıtlarını (.txt) dosyasından dinamik olarak okur.")

# --- SOL MENÜ (BRKGA PARAMETRELERİ) ---
st.sidebar.header("BRKGA Parametreleri")
pop_size = st.sidebar.slider("Popülasyon (p)", 50, 500, 100, 10)
elite_ratio = st.sidebar.slider("Elit Oranı (p_e %)", 5, 40, 20, 5)
mutant_ratio = st.sidebar.slider("Mutant Oranı (p_m %)", 5, 40, 15, 5)
rho_e = st.sidebar.slider("Yanlı Çaprazlama (ρ_e)", 0.50, 0.95, 0.70, 0.05)
max_gen = st.sidebar.number_input("Maksimum Jenerasyon", value=200)

# --- ANA EKRAN (DOSYA YÜKLEME VE ANALİZ) ---
st.subheader("1. Veri Seti Yükle")
uploaded_file = st.file_uploader("txt dosyasını sürükleyin (örn: uniform-51-n10-novisit-30.txt)", type=["txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    
    # 1. Dosyayı Parser'a gönder ve verileri dinamik çöz!
    parsed_data = FSTSP_Parser(content)
    
    st.success("Dosya başarıyla okundu ve ayrıştırıldı!")
    
    # 2. Dinamik olarak okunan değerleri ekrana basarak doğrula
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Düğüm", parsed_data.num_nodes)
    col2.metric("Kamyon Hızı", parsed_data.truck_speed)
    col3.metric("Dron Hızı", parsed_data.drone_speed)
    col4.metric("Dron Batarya (MAXFLY)", "Limitsiz" if parsed_data.max_fly == float('inf') else parsed_data.max_fly)
    
    if len(parsed_data.novisit_list) > 0:
        st.warning(f"⚠️ NOVISIT Kısıtları Bulundu: Dron şu müşterilere gidemez: {parsed_data.novisit_list}")
    else:
        st.info("✅ NOVISIT kısıtı bulunamadı, dron tüm noktalara uçabilir.")

    # 3. Optimizasyon Tetikleyicisi
    st.subheader("2. Optimizasyonu Çalıştır")
    if st.button("🚀 Akıllı Çözücü ile BRKGA'yı Başlat"):
        st.info("Akıllı Çözücü (DAG-Shortest Path) ve BRKGA algoritması buraya entegre edilecek. (Modül 2 ve 3 hazırlanıyor...)")

else:
    st.warning("Lütfen başlamak için bir txt dosyası yükleyin.")

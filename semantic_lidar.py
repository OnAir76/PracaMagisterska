import rclpy
from rclpy.node import Node
import numpy as np
import cv2  
import math
import os  
import struct  
import traceback 
import datetime 

from sensor_msgs.msg import PointCloud2, PointField  
from geometry_msgs.msg import TransformStamped 
from std_msgs.msg import Header 
import tf2_ros
from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA 

try:
    import carla
except ImportError as e:
    print(f"Błąd importu CARLA: {e}")
    exit()

# === STAŁE KONFIGURACYJNE ===
CARLA_HOST = 'localhost'  # Adres IP, na którym działa serwer CARLA
CARLA_PORT = 2000         # Port serwera CARLA
EGO_VEHICLE_ROLE = 'ego_vehicle' # Rola pojazdu, do którego zostanie podłączony Lidar

# Tagi semantyczne zgodnie z dokumentacją CARLA:
# 24: RoadLine - Oznaczenia na drodze (np. linie, strzałki). KLUCZOWY TAG.
INTERESTING_TAGS = {24} # Celowo ograniczono do 'RoadLine' dla precyzji.
LIDAR_Z_MIN = -2.5  # Minimalna wysokość (oś Z) punktów do analizy (odfiltrowanie podłoża)
LIDAR_Z_MAX = -0.5  # Maksymalna wysokość (oś Z) (odfiltrowanie punktów niebędących na jezdni)

# === PARAMETRY ALGORYTMÓW ===
DBSCAN_EPS = 0.4  # Maksymalna odległość między dwoma punktami, aby uznać je za sąsiadów w DBSCAN
DBSCAN_MIN_SAMPLES = 10 # Minimalna liczba punktów, aby utworzyć klaster

MAX_SIGN_DIMENSION_METERS = 10.0  # Maksymalny oczekiwany wymiar znaku w metrach (filtr przeciwko zbyt dużym klastrom)

TEMPLATE_MATCH_THRESHOLD = 0.45 # Próg pewności (wynik z matchTemplate), powyżej którego uznajemy detekcję za poprawną
IMAGE_RESOLUTION = 0.02 # Rozdzielczość obrazu tworzonego z klastra (metry na piksel)

# === USTAWIENIA DEBUGOWANIA ===
SAVE_DEBUG_IMAGES = True  # Czy zapisywać obrazy klastrów i nałożeń do analizy
ENABLE_OVERLAY_ON_DEBUG = True # Czy nakładać dopasowany szablon na obraz klastra dla lepszej wizualizacji

# === ŚCIEŻKI I NAZWY PLIKÓW ===
TEMPLATE_DIR = "/root"  # Katalog, w którym znajdują się obrazy wzorców (np. arrow_left.png)
DEBUG_SAVE_PATH = os.path.join(TEMPLATE_DIR, "tests") # Gdzie zapisywać obrazy debugowania

# Mapowanie nazw klas na kolory (BGR) do wizualizacji w RViz2
COLOR_MAP = {
    "arrow_left": (0, 255, 0), "arrow_right": (0, 255, 255), "stop_template": (255, 0, 0),
    "pedestrian_crossing": (255, 255, 0), "unknown": (255, 0, 255)
}
# Lista plików z szablonami do załadowania
TEMPLATE_FILES = ["arrow_left.png", "arrow_right.png", "stop_template.png", "pedestrian_crossing.png"]
# Mapowanie nazw klas na skrócone nazwy używane w nazwach plików
FILENAME_MAP = {
    "arrow_left": "left", "arrow_right": "right", "stop_template": "stop",
    "pedestrian_crossing": "pasy", "unknown": "nieznany"
}


class SemanticLidarProcessor(Node):
    def __init__(self):
        super().__init__('semantic_lidar_processor_v14_commented')
        # Inicjalizacja zmiennych
        self.client, self.world, self.ego_vehicle, self.lidar_sensor = None, None, None, None
        
        # Publisher do wysyłania przetworzonej, pokolorowanej chmury punktów do RViz2
        self.publisher_ = self.create_publisher(PointCloud2, '/semantic_lidar_colored', 10)
        
        # Broadcaster do wysyłania transformacji położenia Lidaru
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Ładowanie obrazów wzorców z dysku
        self.templates = self.load_templates()
        
        # Tworzenie folderu na obrazy debugowania, jeśli nie istnieje
        if SAVE_DEBUG_IMAGES and not os.path.exists(DEBUG_SAVE_PATH):
            os.makedirs(DEBUG_SAVE_PATH)
            
        # Uruchomienie połączenia z CARLA i konfiguracja sensora
        self.setup_carla()

    def setup_carla(self):
        try:
            self.get_logger().info(f"Łączenie z CARLA pod adresem {CARLA_HOST}:{CARLA_PORT}...")
            self.client = carla.Client(CARLA_HOST, CARLA_PORT)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            
            # Pętla poszukująca pojazdu gracza
            while self.ego_vehicle is None:
                self.get_logger().info("Szukam 'ego_vehicle'...")
                for actor in self.world.get_actors().filter(f"vehicle.*"):
                    if actor.attributes.get('role_name') == EGO_VEHICLE_ROLE:
                        self.ego_vehicle = actor
                        self.get_logger().info("Znaleziono 'ego_vehicle'.")
                        break
                if self.ego_vehicle is None:
                    self.get_logger().warn("Nie znaleziono 'ego_vehicle'. Próbuję ponownie za 2s...")
                    rclpy.spin_once(self, timeout_sec=2.0)

            # Konfiguracja parametrów sensora Lidar
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '50.0') 
            lidar_bp.set_attribute('points_per_second', '1000000') 
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('channels', '64') 
            lidar_bp.set_attribute('lower_fov', '-30.0') 
            lidar_bp.set_attribute('upper_fov', '2.0') 

            # Umieszczenie Lidaru na pojeździe (1.5m w przód, 2.5m w górę)
            lidar_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.5))
            self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_vehicle)
            self.get_logger().info("Semantic LiDAR zamontowany na ego_vehicle.")
            
            # Ustawienie funkcji zwrotnej (callback), która będzie wywoływana dla każdego pomiaru
            self.lidar_sensor.listen(self.lidar_callback)
            
        except Exception as e:
            self.get_logger().error(f"Nie udało się połączyć z CARLA lub skonfigurować sensora: {e}")
            self.destroy_node()

    def load_templates(self):
        templates = {}
        self.get_logger().info(f"Ładowanie szablonów z katalogu: {TEMPLATE_DIR}")
        for filename in TEMPLATE_FILES:
            try:
                path = os.path.join(TEMPLATE_DIR, filename)
                
                template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if template_img is None: raise FileNotFoundError(f"Nie można załadować pliku: {path}")
                
                class_name = os.path.splitext(filename)[0]
                self.get_logger().info(f"Załadowano szablon '{filename}' dla klasy '{class_name}'.")
                templates[class_name] = template_img
            except Exception as e:
                self.get_logger().warn(f"Nie udało się załadować szablonu '{filename}': {e}.")
        if not templates: self.get_logger().warn("UWAGA: Nie załadowano żadnych szablonów! Detekcja nie będzie działać.")
        return templates

    def lidar_callback(self, lidar_data):
        try:
            # 1. PUBLIKACJA TRANSFORMACJI (TF)
            self.publish_tf(lidar_data.transform)

            # 2. DEKODOWANIE DANYCH
            points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
            ]))
            
            # 3. FILTROWANIE PUNKTÓW
            marked_points = points[np.isin(points['ObjTag'], list(INTERESTING_TAGS))]
            road_level_points = marked_points[(marked_points['z'] > LIDAR_Z_MIN) & (marked_points['z'] < LIDAR_Z_MAX)]
            
            # Jeśli po filtrowaniu zostało zbyt mało punktów, przerwij przetwarzanie
            if road_level_points.size < DBSCAN_MIN_SAMPLES:
                self.publish_colored_cloud([]) # Publikuj pustą chmurę
                return

            # 4. KLASTERYZACJA (GRUPOWANIE PUNKTÓW)
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(
                np.array([road_level_points['x'], road_level_points['y']]).T
            )
            unique_labels = set(db.labels_)
            if -1 in unique_labels: unique_labels.remove(-1) # Etykieta -1 oznacza szum (punkty nieprzypisane do klastra)
            
            if not unique_labels: # Jeśli nie znaleziono żadnych klastrów
                self.publish_colored_cloud([])
                return

            final_clusters_info = [] # Lista do przechowywania informacji o zidentyfikowanych znakach
            
            # 5. PRZETWARZANIE KAŻDEGO KLASTRA
            for k in unique_labels:
                cluster_points = road_level_points[db.labels_ == k]

                if len(cluster_points) < DBSCAN_MIN_SAMPLES: continue

                # 6. NORMALIZACJA ORIENTACJI KLASTRA
                pca = PCA(n_components=2).fit(np.array([cluster_points['x'], cluster_points['y']]).T)
                angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0]) # Kąt głównej osi
                
                # Tworzymy macierz rotacji, aby obrócić klaster do pozycji horyzontalnej
                rotation_matrix = np.array([[math.cos(-angle_rad), -math.sin(-angle_rad)], [math.sin(-angle_rad), math.cos(-angle_rad)]])
                rotated_xy = np.dot(np.array([cluster_points['x'], cluster_points['y']]).T, rotation_matrix.T)

                # 7. TWORZENIE OBRAZU Z KLASTRA
                min_coords = np.min(rotated_xy, axis=0)
                max_coords = np.max(rotated_xy, axis=0)
                cluster_size_meters = max_coords - min_coords
                
                if cluster_size_meters[0] > MAX_SIGN_DIMENSION_METERS or cluster_size_meters[1] > MAX_SIGN_DIMENSION_METERS:
                    continue
                
                # przesunięcie punktów do początku układu współrzędnych i przeskalowanie na piksele
                rotated_xy -= min_coords
                pixel_coords = (rotated_xy / IMAGE_RESOLUTION).astype(int)
                
                # Stworzenie pustego, czarnego obrazu o wymiarach klastra
                img_size = np.max(pixel_coords, axis=0) + 1
                if img_size.shape[0] < 2 or img_size[0] < 5 or img_size[1] < 5: continue # Odrzucenie zbyt małych obrazów
                cluster_image = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
                
                # Narysowanie białych pikseli w miejscach, gdzie znajdują się punkty Lidaru
                cluster_image[pixel_coords[:, 1], pixel_coords[:, 0]] = 255
                
                kernel = np.ones((6, 6), np.uint8)
                cluster_image = cv2.dilate(cluster_image, kernel, iterations=1)

                # DOPASOWYWANIE WZORCA 
                best_match_score = -1.0
                best_match_class = "unknown"
                best_template_for_overlay = None
                all_scores_for_this_cluster = {}
                (h_cluster, w_cluster) = cluster_image.shape

                if self.templates:
                    for class_name, template in self.templates.items():
                        best_score_for_this_class = -1.0
                       
                        templates_to_test = [template, cv2.flip(template, -1)]

                        for test_template in templates_to_test:
                            # skalujemy wzorzec do dokładnych wymiarów obrazu klastra
                            resized_template = cv2.resize(test_template, (w_cluster, h_cluster), interpolation=cv2.INTER_AREA)
                            
                            # Porównujemy obraz klastra z przeskalowanym wzorcem
                            res = cv2.matchTemplate(cluster_image, resized_template, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(res)
                            
                            if max_val > best_score_for_this_class: best_score_for_this_class = max_val
                            if max_val > best_match_score:
                                best_match_score = max_val
                                best_match_class = class_name
                                best_template_for_overlay = resized_template
                        
                        all_scores_for_this_cluster[class_name] = best_score_for_this_class
                
                # 9. KLASYFIKACJA KOŃCOWA
                final_class = best_match_class if best_match_score > TEMPLATE_MATCH_THRESHOLD else "unknown"

                # 10. ZAPIS OBRAZÓW DEBUGOWANIA
                if SAVE_DEBUG_IMAGES:
                    canvas = cv2.cvtColor(cluster_image, cv2.COLOR_GRAY2BGR) # Konwersja na obraz kolorowy
                    time_str = datetime.datetime.now().strftime("%H_%M_%S_%f")
                    
                    if ENABLE_OVERLAY_ON_DEBUG and best_template_for_overlay is not None:
                        template_mask = best_template_for_overlay > 0
                        canvas[template_mask] = (0, 255, 0) # Nałożenie zielonej maski wzorca
                        filename = f"{FILENAME_MAP.get(final_class, 'unknown')}_overlay_score_{best_match_score:.2f}_{time_str}.png"
                    else:
                        filename = f"raw_cluster_{FILENAME_MAP.get(final_class, 'unknown')}_score_{best_match_score:.2f}_{time_str}.png"
                    
                    filepath = os.path.join(DEBUG_SAVE_PATH, filename)
                    cv2.imwrite(filepath, canvas)

                # 11. LOGOWANIE WYNIKÓW
                self.get_logger().info(f"--- Klaster {k} (rozmiar: {cluster_size_meters[0]:.1f}x{cluster_size_meters[1]:.1f}m) ---")
                if self.templates:
                    score_details = ", ".join([f"{FILENAME_MAP.get(cn, cn)}: {s:.2f}" for cn, s in sorted(all_scores_for_this_cluster.items())])
                    self.get_logger().info(f"  Szczegółowe wyniki: [ {score_details} ]")
                self.get_logger().info(f"  Zidentyfikowano jako: '{FILENAME_MAP.get(final_class, final_class)}' (najlepszy wynik: {best_match_score:.2f})")
                
                if final_class == "arrow_left":
                    self.get_logger().info("  --> Komunikat specjalny: znaleziono strzalke w lewo")
                elif final_class == "arrow_right":
                    self.get_logger().info("  --> Komunikat specjalny: znaleziono strzalke w prawo")

                # Dodanie informacji o przetworzonym klastrze do listy
                final_clusters_info.append({'class': final_class, 'points': cluster_points})

            # 12. TWORZENIE I PUBLIKACJA POKOLOROWANEJ CHMURY PUNKTÓW
            all_colored_points = []
            for cluster_info in final_clusters_info:
                color_rgb = COLOR_MAP.get(cluster_info['class'], COLOR_MAP["unknown"])
                color_packed = struct.unpack('I', struct.pack('BBBB', color_rgb[2], color_rgb[1], color_rgb[0], 0))[0]
                for point in cluster_info['points']:
                    all_colored_points.append((point['x'], point['y'], point['z'], point['CosAngle'], point['ObjTag'], color_packed))
            self.publish_colored_cloud(all_colored_points)

        except Exception as e:
            self.get_logger().error(f"Wystąpił błąd w lidar_callback: {e}")
            traceback.print_exc()

    def publish_tf(self, transform: 'carla.Transform'):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'  
        t.child_frame_id = 'semantic_lidar_link' 
        
        loc, rot = transform.location, transform.rotation
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = loc.x, -loc.y, loc.z
        
        roll_rad, pitch_rad, yaw_rad = math.radians(rot.roll), -math.radians(rot.pitch), -math.radians(rot.yaw)
        cy, sy = math.cos(yaw_rad * 0.5), math.sin(yaw_rad * 0.5)
        cp, sp = math.cos(pitch_rad * 0.5), math.sin(pitch_rad * 0.5)
        cr, sr = math.cos(roll_rad * 0.5), math.sin(roll_rad * 0.5)
        t.transform.rotation.w = cr*cp*cy + sr*sp*sy
        t.transform.rotation.x = sr*cp*cy - cr*sp*sy
        t.transform.rotation.y = cr*sp*cy + sr*cp*sy
        t.transform.rotation.z = cr*cp*sy - sr*sp*cy
        
        self.tf_broadcaster.sendTransform(t)

    def publish_colored_cloud(self, points_data):
        """
        Konwertuje listę przetworzonych punktów na wiadomość PointCloud2 i ją publikuje.
        """
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='semantic_lidar_link')
        # Definicja pól w chmurze punktów (nazwa, offset, typ danych)
        fields = [PointField(name=n, offset=o*4, datatype=PointField.FLOAT32, count=1) for o,n in enumerate(['x','y','z','intensity'])]
        fields.extend([
            PointField(name='tag', offset=16, datatype=PointField.UINT32, count=1),
            PointField(name='rgb', offset=20, datatype=PointField.UINT32, count=1)
        ])
        point_step = 24  # Rozmiar jednego punktu w bajtach
        num_points = len(points_data)
        data = b'' # Pusty bufor danych
        
        if num_points > 0:
            # Tworzenie tablicy NumPy o zdefiniowanej strukturze i konwersja do bajtów
            structured_array = np.array(points_data, dtype=[
                ('x', np.float32), ('y', np.float32), ('z', np.float32), 
                ('intensity', np.float32), ('tag', np.uint32), ('rgb', np.uint32)
            ])
            data = structured_array.tobytes()
            
        # Złożenie finalnej wiadomości PointCloud2
        cloud_msg = PointCloud2(
            header=header, height=1, width=num_points, is_dense=True, is_bigendian=False,
            fields=fields, point_step=point_step, row_step=point_step * num_points, data=data
        )
        self.publisher_.publish(cloud_msg)

    def destroy_node(self):
        self.get_logger().info("Zamykanie węzła i czyszczenie zasobów CARLA...")
        if self.lidar_sensor and self.lidar_sensor.is_alive:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
            self.get_logger().info("Zatrzymano i zniszczono sensor LiDAR.")
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SemanticLidarProcessor()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError) as e:
        if node and rclpy.ok(): node.get_logger().info(f"Zamykanie węzła z powodu: {type(e).__name__} - {e}")
        else: print(f"Zamykanie z powodu błędu przed pełną inicjalizacją: {e}")
    finally:
        if node and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()

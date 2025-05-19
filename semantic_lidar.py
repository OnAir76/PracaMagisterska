import rclpy  # Biblioteka klienta ROS 2 dla Pythona
from rclpy.node import Node  # Klasa bazowa dla węzłów ROS 2
import carla  # Biblioteka klienta CARLA do interakcji z symulatorem
import time  # Moduł do obsługi czasu, np. pauzowania wykonania
import struct  # Moduł do pakowania i rozpakowywania danych binarnych (dla PointCloud2)
import math  # Moduł matematyczny, np. do funkcji trygonometrycznych (dla TF)

# Importy typów wiadomości ROS 2
from sensor_msgs.msg import PointCloud2, PointField  # Wiadomość chmury punktów i definicja jej pól
from std_msgs.msg import Header  # Standardowy nagłówek wiadomości ROS (m.in. timestamp i frame_id)
from geometry_msgs.msg import TransformStamped  # Wiadomość do publikacji transformacji TF
import tf2_ros  # Narzędzia TF2 dla ROS 2, w tym TransformBroadcaster

# Importy dla type hinting (podpowiedzi typów)
from typing import List, Tuple, Any, Dict

# --- Definicje Tagów Semantycznych CARLA ---
TAG_UNLABELED = 0
TAG_BUILDING = 1
TAG_FENCE = 2
TAG_OTHER = 3  
TAG_PEDESTRIAN = 4  
TAG_POLE = 5  
TAG_ROAD_LINE = 6  # Linie na jezdni (znaki poziome)
TAG_ROAD = 7  
TAG_SIDEWALK = 8  
TAG_VEGETATION = 9  
TAG_VEHICLE = 10  
TAG_WALL = 11  
TAG_TRAFFIC_SIGN = 12  
TAG_SKY = 13  
TAG_GROUND = 14 
TAG_BRIDGE = 15  
TAG_RAILTRACK = 16  
TAG_GUARDRAIL = 17 
TAG_TRAFFIC_LIGHT = 18  
TAG_STATIC = 20  
TAG_DYNAMIC = 21 
TAG_WATER = 22  
TAG_TERRAIN = 23  
TAG_CUSTOM_HORIZONTAL_MARKING = 24 

# Kolor do wyróżnienia znaków poziomych (R, G, B jako float 0.0-1.0)
HIGHLIGHT_ROAD_LINE_COLOR = (1.0, 0.0, 1.0)  

def get_color_for_tag(tag: int) -> int:
    # Mapa tagów semantycznych na kolory
    color_map_rgb_float: Dict[int, Tuple[float, float, float]] = {
        TAG_UNLABELED: (0.0, 0.0, 0.0),         
        TAG_BUILDING: (0.27, 0.27, 0.27),      
        TAG_FENCE: (0.78, 0.63, 0.63),         
        TAG_OTHER: (0.22, 0.35, 0.31),         
        TAG_PEDESTRIAN: (0.86, 0.08, 0.24),   
        TAG_POLE: (0.6, 0.6, 0.6),            
        TAG_ROAD_LINE: HIGHLIGHT_ROAD_LINE_COLOR, # Wyróżniony MAGENTA dla linii drogowych
        TAG_ROAD: (0.5, 0.25, 0.5),            
        TAG_SIDEWALK: (0.96, 0.14, 0.91),      
        TAG_VEGETATION: (0.42, 0.56, 0.14),    
        TAG_VEHICLE: (0.0, 0.0, 0.56),         
        TAG_WALL: (0.4, 0.4, 0.61),            
        TAG_TRAFFIC_SIGN: (0.86, 0.86, 0.0),   
        TAG_SKY: (0.27, 0.51, 0.71),          
        TAG_GROUND: (0.32, 0.0, 0.32),         
        TAG_BRIDGE: (0.59, 0.39, 0.39),        
        TAG_RAILTRACK: (0.9, 0.59, 0.55),      
        TAG_GUARDRAIL: (0.71, 0.65, 0.71),     
        TAG_TRAFFIC_LIGHT: (0.98, 0.67, 0.12), 
        TAG_STATIC: (0.43, 0.75, 0.63),        
        TAG_DYNAMIC: (0.67, 0.47, 0.2),        
        TAG_WATER: (0.18, 0.24, 0.59),         
        TAG_TERRAIN: (0.57, 0.67, 0.39),       
        TAG_CUSTOM_HORIZONTAL_MARKING: HIGHLIGHT_ROAD_LINE_COLOR, 
    }

    # Pobierz kolor z mapy. Jeśli tag nie istnieje w mapie, użyj domyślnego koloru (biały).
    r_float, g_float, b_float = color_map_rgb_float.get(tag, (1.0, 1.0, 1.0)) 

    # Konwersja wartości float (0.0-1.0) na int (0-255)
    r_int = int(r_float * 255)
    g_int = int(g_float * 255)
    b_int = int(b_float * 255)
    
    # Spakowanie wartości R, G, B do pojedynczej 32-bitowej liczby całkowitej
    rgb_packed = (r_int << 16) | (g_int << 8) | b_int
    return rgb_packed


def create_point_cloud(points: List[List[Any]], stamp: Any, frame_id: str = "semantic_lidar") -> PointCloud2:
    # Utwórz nagłówek wiadomości
    header = Header()
    header.stamp = stamp  # Czas wygenerowania danych
    header.frame_id = frame_id  # Ramka odniesienia, w której są współrzędne punktów

    # Każdy punkt będzie miał współrzędne x, y, z, intensywność, tag semantyczny i kolor RGB.
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1), # Współrzędna X
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1), # Współrzędna Y
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1), # Współrzędna Z
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1), # Intensywność 
        PointField(name="tag", offset=16, datatype=PointField.UINT32, count=1),       # Tag semantyczny obiektu
        PointField(name="rgb", offset=20, datatype=PointField.UINT32, count=1),       

    point_step = 24  # Rozmiar jednego punktu w bajtach 
    data_buffer = bytearray()  # Bufor na binarne dane punktów

    # Przetwórz każdy punkt z listy wejściowej
    for p_data in points:
        object_tag = int(round(p_data[4]))  # Zaokrąglij i przekonwertuj tag na int
        rgb_color = get_color_for_tag(object_tag)  # Pobierz spakowany kolor dla tagu

        packed_point = struct.pack("ffffII",
                                   float(p_data[0]), float(p_data[1]), float(p_data[2]), # x, y, z
                                   float(p_data[3]),  # intensity
                                   object_tag,        # tag
                                   rgb_color)         # rgb
        data_buffer.extend(packed_point)  # Dodaj spakowany punkt do bufora

    # Utwórz i wypełnij obiekt wiadomości PointCloud2
    pointcloud_msg = PointCloud2()
    pointcloud_msg.header = header
    pointcloud_msg.height = 1  # Chmura nieuporządkowana (jedna linia punktów)
    pointcloud_msg.width = len(points)  # Liczba punktów w chmurze
    pointcloud_msg.fields = fields  # Definicja pól
    pointcloud_msg.is_bigendian = False  # Porządek bajtów (zazwyczaj little-endian)
    pointcloud_msg.point_step = point_step  # Długość jednego punktu w bajtach
    pointcloud_msg.row_step = point_step * pointcloud_msg.width  # Całkowita długość danych w bajtach
    pointcloud_msg.is_dense = True  # Czy chmura zawiera nieprawidłowe punkty Zakładamy, że nie.
    pointcloud_msg.data = bytes(data_buffer)  # Surowe dane binarne chmury punktów
    
    return pointcloud_msg


class SemanticLidarPublisher(Node):
    def __init__(self):
        super().__init__("semantic_lidar_publisher_node") # Inicjalizacja klasy bazowej Node

        # Deklaracja i pobieranie parametrów ROS 2 (umożliwia konfigurację węzła z zewnątrz)
        self.carla_host = self.declare_parameter("carla_host", "localhost").get_parameter_value().string_value
        self.carla_port = self.declare_parameter("carla_port", 2000).get_parameter_value().integer_value
        self.carla_timeout = self.declare_parameter("carla_timeout", 10.0).get_parameter_value().double_value
        self.tf_broadcast_rate = self.declare_parameter("tf_broadcast_rate", 0.1).get_parameter_value().double_value # Częstotliwość publikacji TF (w sekundach)
        self.lidar_topic = self.declare_parameter("lidar_topic", "/carla/ego_vehicle/semantic_lidar").get_parameter_value().string_value
        self.map_frame = self.declare_parameter("map_frame", "map").get_parameter_value().string_value # Nazwa ramki świata
        self.lidar_frame = self.declare_parameter("lidar_frame", "semantic_lidar").get_parameter_value().string_value # Nazwa ramki LiDARa
        self.ego_vehicle_role_name = self.declare_parameter("ego_vehicle_role_name", "ego_vehicle").get_parameter_value().string_value # Rola pojazdu w CARLA
        self.log_detected_tags = self.declare_parameter("log_detected_tags", True).get_parameter_value().bool_value # Czy logować wykryte tagi?

        # Parametry konfiguracyjne LiDARa (pobierane jako parametry ROS 2)
        self.lidar_channels = self.declare_parameter("lidar.channels", "64").get_parameter_value().string_value
        self.lidar_range = self.declare_parameter("lidar.range", "120").get_parameter_value().string_value 
        self.lidar_points_per_second = self.declare_parameter("lidar.points_per_second", "1000000").get_parameter_value().string_value
        self.lidar_rotation_frequency = self.declare_parameter("lidar.rotation_frequency", "20").get_parameter_value().string_value
        self.lidar_upper_fov = self.declare_parameter("lidar.upper_fov", "10").get_parameter_value().string_value
        self.lidar_lower_fov = self.declare_parameter("lidar.lower_fov", "-35").get_parameter_value().string_value
        
        # Parametry transformacji LiDARa względem pojazdu ego
        self.lidar_pos_x = self.declare_parameter("lidar.transform.x", 0.0).get_parameter_value().double_value
        self.lidar_pos_y = self.declare_parameter("lidar.transform.y", 0.0).get_parameter_value().double_value
        self.lidar_pos_z = self.declare_parameter("lidar.transform.z", 2.5).get_parameter_value().double_value 
        self.lidar_rot_pitch = self.declare_parameter("lidar.transform.pitch", 0.0).get_parameter_value().double_value 
        self.lidar_rot_yaw = self.declare_parameter("lidar.transform.yaw", 0.0).get_parameter_value().double_value
        self.lidar_rot_roll = self.declare_parameter("lidar.transform.roll", 0.0).get_parameter_value().double_value

        # Utworzenie publishera dla wiadomości PointCloud2
        self.publisher = self.create_publisher(PointCloud2, self.lidar_topic, 10) # 10 to rozmiar kolejki
        # Utworzenie broadcastera dla transformacji TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Utworzenie timera do regularnego publikowania transformacji TF
        self.timer_tf = self.create_timer(self.tf_broadcast_rate, self.broadcast_tf_transform)

        # Inicjalizacja zmiennych członkowskich
        self.client: carla.Client = None
        self.world: carla.World = None
        self.ego_vehicle: carla.Actor = None # Aktor pojazdu ego
        self.sensor: carla.Actor = None      # Aktor sensora LiDAR

        # Próba połączenia z CARLA i skonfigurowania sensora
        try:
            if not self.connect_to_carla():
                raise RuntimeError("Nie udało się połączyć z CARLA.")
            if not self.find_ego_vehicle():
                raise RuntimeError(f"Nie znaleziono pojazdu z rolą '{self.ego_vehicle_role_name}'.")
            self.spawn_lidar()
        except RuntimeError as e:
            self.get_logger().error(f"Krytyczny błąd podczas inicjalizacji: {e} Węzeł zostanie zamknięty.")
            # Anuluj timer, jeśli został utworzony, przed rzuceniem wyjątku
            if hasattr(self, 'timer_tf') and self.timer_tf: 
                self.timer_tf.cancel()
            raise # Rzuć wyjątek dalej, aby main mógł go obsłużyć

    def connect_to_carla(self) -> bool:
        self.get_logger().info(f"Łączenie z serwerem CARLA: {self.carla_host}:{self.carla_port}...")
        try:
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(self.carla_timeout) # Ustawienie timeoutu dla operacji klienta
            self.world = self.client.get_world() # Pobranie referencji do świata symulacji
            # Sprawdzenie połączenia przez pobranie wersji serwera
            self.get_logger().info(f"Połączono z CARLA (wersja serwera: {self.client.get_server_version()}, wersja klienta: {self.client.get_client_version()}).")
            return True
        except RuntimeError as e:
            self.get_logger().error(f"Nie można połączyć z CARLA: {e}")
            return False

    def find_ego_vehicle(self) -> bool:
        self.get_logger().info(f"Wyszukiwanie pojazdu ego: '{self.ego_vehicle_role_name}'...")
        max_retries = 10 # Maksymalna liczba prób znalezienia pojazdu
        for i in range(max_retries):
            actors = self.world.get_actors().filter("vehicle.*") # Pobierz wszystkich aktorów typu pojazd
            for actor in actors:
                # Sprawdź, czy aktor ma atrybut 'role_name' i czy jest on równy poszukiwanej roli
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == self.ego_vehicle_role_name:
                    self.ego_vehicle = actor
                    self.get_logger().info(f"Znaleziono pojazd ego: {self.ego_vehicle.type_id} (ID: {self.ego_vehicle.id})")
                    return True
            self.get_logger().warn(f"Pojazd ego nie znaleziony, próba {i+1}/{max_retries}...")
            time.sleep(1.0) # Poczekaj sekundę przed kolejną próbą
        return False

    def spawn_lidar(self):
        if not self.ego_vehicle: 
            self.get_logger().error("Pojazd ego nie jest dostępny, nie można utworzyć LiDARa.")
            raise RuntimeError("Pojazd ego nie jest dostępny dla LiDARa.")

        self.get_logger().info("Tworzenie sensora Semantic LiDAR...")
        # Logowanie używanych parametrów LiDARa dla łatwiejszego debugowania
        self.get_logger().info(f"  Używane parametry LiDARa: range={self.lidar_range}, points_per_second={self.lidar_points_per_second}, lower_fov={self.lidar_lower_fov}, pitch={self.lidar_rot_pitch}")
        try:
            blueprint_library = self.world.get_blueprint_library() # Pobierz bibliotekę blueprintów
            bp = blueprint_library.find("sensor.lidar.ray_cast_semantic") # Znajdź blueprint semantycznego LiDARa

            # Ustawianie atrybutów LiDARa na podstawie pobranych parametrów
            bp.set_attribute("channels", self.lidar_channels)
            bp.set_attribute("range", self.lidar_range)
            bp.set_attribute("points_per_second", self.lidar_points_per_second)
            bp.set_attribute("rotation_frequency", self.lidar_rotation_frequency)
            bp.set_attribute("upper_fov", self.lidar_upper_fov)
            bp.set_attribute("lower_fov", self.lidar_lower_fov)
            
            # Definicja transformacji LiDARa względem pojazdu ego
            # (pozycja i orientacja sensora na pojeździe)
            transform = carla.Transform(
                carla.Location(x=self.lidar_pos_x, y=self.lidar_pos_y, z=self.lidar_pos_z),
                carla.Rotation(pitch=self.lidar_rot_pitch, yaw=self.lidar_rot_yaw, roll=self.lidar_rot_roll)
            )

            # Utworzenie aktora sensora w świecie CARLA i podpięcie go do pojazdu ego
            self.sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego_vehicle)
            # Rozpoczęcie nasłuchiwania danych z sensora; funkcja 'lidar_callback' będzie wywoływana przy każdym nowym odczycie
            self.sensor.listen(self.lidar_callback)
            self.get_logger().info(f"Semantic LiDAR (ID: {self.sensor.id}) utworzony i nasłuchuje.")
        except RuntimeError as e:
            self.get_logger().error(f"Błąd podczas tworzenia LiDARa: {e}")
            # Jeśli sensor został częściowo utworzony, spróbuj go zniszczyć
            if self.sensor and self.sensor.is_alive: 
                self.sensor.destroy()
            self.sensor = None
            raise # Rzuć wyjątek dalej

    def lidar_callback(self, carla_data: carla.SemanticLidarMeasurement):
        """
        Funkcja zwrotna (callback) wywoływana przy każdym nowym odczycie danych z LiDARa.
        Przetwarza dane i publikuje je jako wiadomość PointCloud2.
        """
        points_data: List[List[Any]] = [] # Lista do przechowywania przetworzonych punktów
        for detection in carla_data:
            # Dla każdej detekcji (punktu) pobierz potrzebne informacje
            points_data.append([
                detection.point.x,         # Współrzędna X punktu (w układzie sensora)
                detection.point.y,         # Współrzędna Y punktu
                detection.point.z,         # Współrzędna Z punktu
                detection.cos_inc_angle,   # Cosinus kąta padania (może służyć jako intensywność)
                detection.object_tag       # Tag semantyczny obiektu, do którego należy punkt
            ])

        if points_data: # Jeśli zebrano jakieś punkty
            if self.log_detected_tags: # Jeśli włączono logowanie tagów
                # Zbierz unikalne tagi z bieżącej ramki danych
                detected_tags = set(int(round(p[4])) for p in points_data)
                if detected_tags: 
                    # Loguj posortowaną listę unikalnych tagów (ograniczone logowanie, aby nie zalać konsoli)
                    self.get_logger().info(f"Wykryte tagi w tej ramce: {sorted(list(detected_tags))}",
                                           throttle_duration_sec=5.0) 
                                           
            stamp = self.get_clock().now().to_msg() # Pobierz aktualny czas ROS
            # Utwórz wiadomość PointCloud2; dane są w ramce LiDARa
            pc2_msg = create_point_cloud(points_data, stamp, frame_id=self.lidar_frame)
            self.publisher.publish(pc2_msg) # Opublikuj wiadomość

    def broadcast_tf_transform(self):
        # Sprawdź, czy sensor istnieje i jest aktywny
        if not self.sensor or not self.sensor.is_alive:
            return

        try:
            # Pobierz aktualną globalną transformację sensora LiDAR w świecie CARLA
            carla_sensor_transform_world: carla.Transform = self.sensor.get_transform()
            loc = carla_sensor_transform_world.location # Pozycja (x, y, z)
            rot = carla_sensor_transform_world.rotation # Orientacja (pitch, yaw, roll)

            # Utwórz wiadomość TransformStamped
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg() # Czas transformacji
            t.header.frame_id = self.map_frame  # Ramka nadrzędna (świat)
            t.child_frame_id = self.lidar_frame # Ramka potomna (LiDAR)

            # Ustaw translację (pozycję) LiDARa w ramce mapy
            t.transform.translation.x = loc.x
            t.transform.translation.y = -loc.y # Odwrócenie osi Y
            t.transform.translation.z = loc.z

            # Ustaw rotację (orientację) LiDARa w ramce mapy
            roll_rad = math.radians(rot.roll)
            pitch_rad = -math.radians(rot.pitch) # Odwrócenie znaku dla pitch
            yaw_rad = -math.radians(rot.yaw)     # Odwrócenie znaku dla yaw

            cy = math.cos(yaw_rad * 0.5)
            sy = math.sin(yaw_rad * 0.5)
            cp = math.cos(pitch_rad * 0.5)
            sp = math.sin(pitch_rad * 0.5)
            cr = math.cos(roll_rad * 0.5)
            sr = math.sin(roll_rad * 0.5)

            t.transform.rotation.w = cr * cp * cy + sr * sp * sy
            t.transform.rotation.x = sr * cp * cy - cr * sp * sy
            t.transform.rotation.y = cr * sp * cy + sr * cp * sy
            t.transform.rotation.z = cr * cp * sy - sr * sp * cy

            # Wyślij transformację
            self.tf_broadcaster.sendTransform(t)
        except RuntimeError as e: # Częsty błąd, jeśli sensor jest niszczony w międzyczasie
            self.get_logger().warn(f"Nie udało się pobrać transformacji sensora dla TF: {e}", throttle_duration_sec=5.0)
        except Exception as e: # Inne nieoczekiwane błędy
            self.get_logger().error(f"Niespodziewany błąd w broadcast_tf_transform: {type(e).__name__} - {e}", throttle_duration_sec=5.0)

    def destroy_node_resources(self):
        self.get_logger().info("Zwalnianie zasobów węzła...")
        if self.sensor: # Jeśli sensor został utworzony
            if self.sensor.is_listening: # Jeśli sensor nasłuchuje
                self.get_logger().info(f"Zatrzymywanie nasłuchu sensora LiDAR (ID: {self.sensor.id})...")
                self.sensor.stop()
            if self.sensor.is_alive: # Jeśli aktor sensora nadal istnieje w symulacji
                self.get_logger().info(f"Niszczenie aktora sensora LiDAR (ID: {self.sensor.id})...")
                if self.sensor.destroy(): # Spróbuj zniszczyć aktora
                    self.get_logger().info(f"Sensor LiDAR (ID: {self.sensor.id}) zniszczony pomyślnie.")
                else:
                    self.get_logger().warn(f"Nie udało się zniszczyć sensora LiDAR (ID: {self.sensor.id}). Mógł już zostać zniszczony.")
            self.sensor = None # Usuń referencję

def main(args=None):
    """Główna funkcja uruchamiająca węzeł ROS 2."""
    rclpy.init(args=args) # Inicjalizacja systemu ROS 2
    node = None # Inicjalizacja zmiennej node
    try:
        node = SemanticLidarPublisher() # Utworzenie instancji węzła
        rclpy.spin(node) # Rozpoczęcie pętli zdarzeń ROS 2 (utrzymuje węzeł przy życiu i przetwarza callbacki)
    except KeyboardInterrupt: # Obsługa przerwania przez użytkownika (Ctrl+C)
        if node: node.get_logger().info("Przechwycono KeyboardInterrupt (Ctrl+C).")
        else: print("Przechwycono KeyboardInterrupt (Ctrl+C) podczas inicjalizacji.")
    except RuntimeError as e: # Obsługa błędów wykonania (np. z __init__)
        if node: node.get_logger().fatal(f"Błąd wykonania podczas inicjalizacji uniemożliwił uruchomienie: {e}")
        else: print(f"Błąd wykonania podczas inicjalizacji uniemożliwił uruchomienie: {e}")
    except Exception as e: # Obsługa innych nieoczekiwanych błędów
        if node: node.get_logger().fatal(f"Nieoczekiwany błąd globalny: {type(e).__name__} - {e}")
        else: print(f"Nieoczekiwany błąd globalny podczas inicjalizacji: {type(e).__name__} - {e}")
    finally:
        # Sekcja wykonywana zawsze na końcu (nawet jeśli wystąpił błąd lub przerwanie)
        if node:
            node.get_logger().info("Rozpoczynanie zamykania węzła...")
            node.destroy_node_resources() # Zwolnienie zasobów CARLA
            if hasattr(node, 'timer_tf') and node.timer_tf: # Anulowanie timera TF, jeśli istnieje
                node.timer_tf.cancel()
            if not node.is_shutdown(): # Sprawdzenie, czy węzeł nie został już zamknięty
                 node.destroy_node() # Zniszczenie węzła ROS 2
            node.get_logger().info("Węzeł zakończył działanie.")
        if rclpy.ok(): # Sprawdzenie, czy system ROS 2 jest nadal aktywny
            rclpy.shutdown() # Zamknięcie systemu ROS 2
        print("Aplikacja zakończona.")

if __name__ == "__main__":
    main() 
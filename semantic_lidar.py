import rclpy
from rclpy.node import Node
import carla
import time
import struct
import math
from typing import List, Tuple, Any, Optional, Dict

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
import tf2_ros

import numpy as np
from sklearn.cluster import DBSCAN

# --- Tagi semantyczne CARLA ---
TAG_UNLABELED = 0
TAG_BUILDING = 1
TAG_FENCE = 2
TAG_OTHER = 3
TAG_PEDESTRIAN = 4
TAG_POLE = 5
TAG_ROAD_LINE = 6
TAG_ROAD = 7
TAG_SIDEWALK = 8
TAG_VEGETATION = 9
TAG_VEHICLE = 10
TAG_WALL = 11
TAG_TRAFFIC_SIGN = 12
TAG_SKY = 13
TAG_GROUND = 14
TAG_BRIDGE = 15
TAG_RAIL_TRACK = 16
TAG_GUARD_RAIL = 17
TAG_TRAFFIC_LIGHT = 18
TAG_STATIC = 19
TAG_DYNAMIC = 20
TAG_WATER = 21
TAG_TERRAIN = 22
TAG_CUSTOM_HORIZONTAL_MARKING = 24

# --- Definicje kolorów (R, G, B) ---
COLOR_STOP_LINE = (1.0, 0.0, 0.0)
COLOR_ARROW_LEFT = (0.0, 1.0, 0.0)
COLOR_ARROW_RIGHT = (0.0, 0.0, 1.0)
COLOR_ARROW_STRAIGHT = (1.0, 1.0, 0.0)
COLOR_PEDESTRIAN_CROSSING = (0.0, 1.0, 1.0)
COLOR_ROAD_LINE = (1.0, 1.0, 1.0)
COLOR_OTHER_CUSTOM_MARKING = (1.0, 0.0, 1.0)
COLOR_DEFAULT = (0.5, 0.5, 0.5)

LIDAR_MOUNT_HEIGHT_ABOVE_GROUND = 2.5
MIN_Z_RELATIVE_TO_SENSOR = -LIDAR_MOUNT_HEIGHT_ABOVE_GROUND - 0.5
MAX_Z_RELATIVE_TO_SENSOR = -LIDAR_MOUNT_HEIGHT_ABOVE_GROUND + 1.0

def classify_cluster(cluster_points_xy: np.ndarray, all_cluster_points: List[List[Any]], logger: Optional[Any] = None) -> str:
    num_points = len(all_cluster_points)

    if num_points < 5:
        return "unknown"

    x_coords = cluster_points_xy[:, 0]
    y_coords = cluster_points_xy[:, 1]
    x_range = np.ptp(x_coords) if x_coords.size > 0 else 0
    y_range = np.ptp(y_coords) if y_coords.size > 0 else 0

    if x_range < 1.0 and y_range > 1.5 and num_points > 20:
        return "stop_line"

    if x_range > 1.5 and y_range > 1.8 and num_points > 50:
        return "pedestrian_crossing"

    if x_range > 1.0 and y_range < (x_range * 0.7) and (x_range / (y_range + 1e-6)) > 1.5 and num_points > 15: 
        return "arrow_straight"

    aspect_ratio_xy = x_range / (y_range + 1e-6)

    if y_range > 0.6 and x_range < (y_range * 0.9) and aspect_ratio_xy < 0.8 and num_points > 10: 
        return "arrow_left"

    if x_range > 0.6 and y_range < (x_range * 0.9) and aspect_ratio_xy > 1.1 and num_points > 10: 
        return "arrow_right"

    # Próba rozpoznania linii ciągłej (jeśli CARLA dała jej tag 24)
    if (x_range > 3.0 and y_range < 0.7 and num_points > 25) or \
       (y_range > 3.0 and x_range < 0.7 and num_points > 25):
       return "continuous_line_custom"

    return "unknown"

def get_color_for_tag(tag: int, mark_type: Optional[str] = None) -> Tuple[float, float, float]:
    if tag == TAG_CUSTOM_HORIZONTAL_MARKING:
        if mark_type == "stop_line":
            return COLOR_STOP_LINE
        elif mark_type == "arrow_left":
            return COLOR_ARROW_LEFT
        elif mark_type == "arrow_right":
            return COLOR_ARROW_RIGHT
        elif mark_type == "arrow_straight":
            return COLOR_ARROW_STRAIGHT
        elif mark_type == "pedestrian_crossing":
            return COLOR_PEDESTRIAN_CROSSING
        elif mark_type == "continuous_line_custom":
             return COLOR_ROAD_LINE 
        else: 
            return COLOR_OTHER_CUSTOM_MARKING
    elif tag == TAG_ROAD_LINE:
        return COLOR_ROAD_LINE
    else:
        return COLOR_DEFAULT

def create_point_cloud(points: List[List[Any]], cluster_labels: Dict[int, str], stamp: Any, frame_id: str) -> Optional[PointCloud2]:
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="tag", offset=16, datatype=PointField.UINT32, count=1),
        PointField(name="rgb", offset=20, datatype=PointField.UINT32, count=1),
    ]
    data_buffer = bytearray()
    for p_record in points:
        tag = int(round(p_record[4]))
        mark_type = None
        if tag == TAG_CUSTOM_HORIZONTAL_MARKING:
            mark_type = cluster_labels.get(id(p_record), None)
        
        color_rgb_float = get_color_for_tag(tag, mark_type)

        r_int = int(color_rgb_float[0] * 255)
        g_int = int(color_rgb_float[1] * 255)
        b_int = int(color_rgb_float[2] * 255)
        rgb_packed = (r_int << 16) | (g_int << 8) | b_int
        
        packed_point = struct.pack("ffffII", 
                                   float(p_record[0]), 
                                   float(p_record[1]), 
                                   float(p_record[2]), 
                                   float(p_record[3]),
                                   tag,
                                   rgb_packed)
        data_buffer.extend(packed_point)

    if not data_buffer:
        return None
        
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = bytes(data_buffer)
    return msg

class SemanticLidarPublisher(Node):
    def __init__(self):
        super().__init__("semantic_lidar_classifier")
        self.publisher = self.create_publisher(PointCloud2, "/semantic_lidar_colored", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer_tf = self.create_timer(0.1, self.broadcast_tf_transform)
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        
        self.get_logger().info("Oczekiwanie na połączenie ze światem CARLA i pojawienie się pojazdu...")
        self.ego_vehicle = None
        for i in range(20):
            self.ego_vehicle = self.get_ego_vehicle("ego_vehicle")
            if self.ego_vehicle:
                break
            self.get_logger().info(f"Pojazd 'ego_vehicle' nie znaleziony, próba {i+1}/20...")
            time.sleep(1)
        
        if not self.ego_vehicle:
            self.get_logger().error("Nie udało się znaleźć pojazdu 'ego_vehicle'. Zamykanie.")
            raise RuntimeError("Ego vehicle 'ego_vehicle' not found")
            
        self.get_logger().info(f"Znaleziono pojazd ego: {self.ego_vehicle.type_id}")
        self.sensor = self.spawn_lidar()
        if not self.sensor:
            self.get_logger().error("Nie udało się utworzyć sensora LiDAR. Zamykanie.")
            rclpy.shutdown()
            return

    def get_ego_vehicle(self, role_name: str) -> Optional[carla.Actor]:
        vehicles = self.world.get_actors().filter("vehicle.*")
        for actor in vehicles:
            if actor.attributes.get("role_name") == role_name:
                return actor
        return None

    def spawn_lidar(self) -> Optional[carla.Actor]:
        if not self.ego_vehicle:
            self.get_logger().error("Nie można utworzyć LiDARa, brak pojazdu ego.")
            return None
        try:
            bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
            bp.set_attribute("channels", "128")
            bp.set_attribute("range", "120")
            bp.set_attribute("points_per_second", "1000000")
            bp.set_attribute("rotation_frequency", "20")
            bp.set_attribute("upper_fov", "30")
            bp.set_attribute("lower_fov", "-30")
            transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=LIDAR_MOUNT_HEIGHT_ABOVE_GROUND))
            
            sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego_vehicle)
            sensor.listen(self.lidar_callback)
            self.get_logger().info("Semantic LiDAR uruchomiony i nasłuchuje.")
            return sensor
        except Exception as e:
            self.get_logger().error(f"Błąd podczas tworzenia LiDARa: {e}")
            return None

    def lidar_callback(self, data: carla.SemanticLidarMeasurement):
        all_points = []
        marking_points_for_clustering = []
        original_marking_point_records = []

        for p_carla in data:
            if MIN_Z_RELATIVE_TO_SENSOR <= p_carla.point.z <= MAX_Z_RELATIVE_TO_SENSOR:
                record = [p_carla.point.x, p_carla.point.y, p_carla.point.z, 
                          p_carla.cos_inc_angle, p_carla.object_tag]
                all_points.append(record)
                
                if p_carla.object_tag == TAG_CUSTOM_HORIZONTAL_MARKING:
                    marking_points_for_clustering.append([p_carla.point.x, p_carla.point.y])
                    original_marking_point_records.append(record)

        cluster_labels_map = {}

        if original_marking_point_records and marking_points_for_clustering:
            coords_np = np.array(marking_points_for_clustering)
            # Parametry DBSCAN
            db = DBSCAN(eps=0.8, min_samples=6).fit(coords_np) 
            labels = db.labels_
            
            unique_labels = set(labels)

            for label_id in unique_labels:
                if label_id == -1:
                    continue
                
                current_cluster_original_records = []
                current_cluster_xy_coords = []
                
                for i, point_label in enumerate(labels):
                    if point_label == label_id:
                        current_cluster_original_records.append(original_marking_point_records[i])
                        current_cluster_xy_coords.append(marking_points_for_clustering[i])
                
                if not current_cluster_original_records:
                    continue

                cluster_xy_np = np.array(current_cluster_xy_coords)
                num_pts_klastra = len(current_cluster_original_records)
                
                mark_type = classify_cluster(cluster_xy_np, current_cluster_original_records, self.get_logger()) 
                
                if mark_type != "unknown":
                    log_messages_map = {
                        "stop_line": "Znaleziono: Linię stopu",
                        "arrow_left": "Znaleziono: Strzałkę w lewo",
                        "arrow_right": "Znaleziono: Strzałkę w prawo",
                        "arrow_straight": "Znaleziono: Strzałkę prosto",
                        "pedestrian_crossing": "Znaleziono: Pasy dla pieszych",
                        "continuous_line_custom": "Znaleziono: Linię ciągłą (z tagu 24)"
                    }
                    message = log_messages_map.get(mark_type, f"Znaleziono: Inny znak poziomy (typ='{mark_type}')")
                    self.get_logger().info(f"{message}")


                for p_record_in_cluster in current_cluster_original_records:
                    cluster_labels_map[id(p_record_in_cluster)] = mark_type
        
        point_cloud_msg = create_point_cloud(all_points, 
                                             cluster_labels_map, 
                                             self.get_clock().now().to_msg(), 
                                             "semantic_lidar_link")
        if point_cloud_msg:
            self.publisher.publish(point_cloud_msg)

    def broadcast_tf_transform(self):
        if not self.sensor:
            return
        
        try:
            carla_transform = self.sensor.get_transform()
        except RuntimeError as e:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map" 
        t.child_frame_id = "semantic_lidar_link"

        t.transform.translation.x = carla_transform.location.x
        t.transform.translation.y = -carla_transform.location.y
        t.transform.translation.z = carla_transform.location.z

        roll_rad = math.radians(carla_transform.rotation.roll)
        pitch_rad = -math.radians(carla_transform.rotation.pitch)
        yaw_rad = -math.radians(carla_transform.rotation.yaw)

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
        
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SemanticLidarPublisher()
        if node and node.sensor:
            rclpy.spin(node)
        else:
            if node:
                node.get_logger().error("Nie udało się w pełni zainicjować węzła SemanticLidarPublisher.")
    except KeyboardInterrupt:
        if node: node.get_logger().info("Zatrzymano przez użytkownika (KeyboardInterrupt).")
    except RuntimeError as e:
        if node: node.get_logger().error(f"Błąd wykonania: {e}")
    finally:
        if node:
            if hasattr(node, 'sensor') and node.sensor:
                try:
                    if node.sensor.is_listening:
                         node.sensor.stop()
                    if hasattr(node.sensor, 'is_destroyed') and not node.sensor.is_destroyed: 
                        node.sensor.destroy()
                    elif not hasattr(node.sensor, 'is_destroyed'): 
                        node.sensor.destroy()
                except Exception as e:
                    node.get_logger().error(f"Błąd podczas niszczenia sensora: {e}")
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

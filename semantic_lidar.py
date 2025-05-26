#!/usr/bin/env python3

# Importy ROS2, Carla oraz narzędzi matematycznych
import rclpy
from rclpy.node import Node
import carla
import time
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
import tf2_ros
import math
import numpy as np
from sklearn.cluster import DBSCAN

# Semantyczny tag Carla dla oznakowania poziomego
TAG_CUSTOM_HORIZONTAL_MARKING = 24

# Mapa kolorów przypisanych do typów oznakowania
COLOR_MAP = {
    "stop_line": (1.0, 0.0, 0.0),             # czerwony
    "arrow_left": (0.0, 1.0, 0.0),             # zielony
    "arrow_right": (0.0, 0.0, 1.0),            # niebieski
    "pedestrian_crossing": (0.0, 1.0, 1.0),    # błękitny
    "unknown": (1.0, 0.0, 1.0),                # różowy
}

# Filtr wysokości — uwzględniamy punkty na poziomie jezdni
LIDAR_MOUNT_HEIGHT = 1.0
MIN_Z_REL = -LIDAR_MOUNT_HEIGHT
MAX_Z_REL = -LIDAR_MOUNT_HEIGHT + 0.5

# Funkcja pomocnicza do pakowania koloru w format uint32

def pack_rgb(r, g, b):
    r_int = round(r * 255)
    g_int = round(g * 255)
    b_int = round(b * 255)
    return (r_int << 16) | (g_int << 8) | b_int

# Funkcja klasyfikująca kształt klastra oznakowania poziomego

def classify_cluster(cluster_xy):
    x_range = np.ptp(cluster_xy[:, 0])
    y_range = np.ptp(cluster_xy[:, 1])
    aspect = x_range / (y_range + 1e-6)
    num_points = len(cluster_xy)

    if x_range < 1.0 and y_range > 1.5 and num_points > 20:
        return "stop_line"
    if x_range > 1.5 and y_range > 1.8 and num_points > 40:
        return "pedestrian_crossing"
    if y_range > 0.6 and x_range < (y_range * 0.9) and aspect < 0.8 and num_points > 10:
        return "arrow_left"
    if x_range > 0.6 and y_range < (x_range * 0.9) and aspect > 1.1 and num_points > 10:
        return "arrow_right"
    return "unknown"

# Główna klasa węzła ROS2
class SemanticLidarShowTags(Node):
    def __init__(self):
        super().__init__('semantic_lidar_show_tags')
        self.publisher = self.create_publisher(PointCloud2, '/semantic_lidar_colored', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensor = None
        self.connect_carla()

        # Sprawdzenie czy znaleziono pojazd ego
        if not self.ego_vehicle:
            self.get_logger().error('Nie znaleziono pojazdu ego!')
            rclpy.shutdown()
            return

        # Inicjalizacja sensora LiDAR
        self.sensor = self.spawn_lidar()
        if not self.sensor:
            self.get_logger().error('Nie udało się zainicjować sensora!')
            rclpy.shutdown()
            return

        # Timer do wysyłania transformacji TF
        self.create_timer(0.1, self.broadcast_tf)

    # Nawiązanie połączenia z Carla i odnalezienie pojazdu
    def connect_carla(self):
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            for _ in range(30):
                for actor in self.world.get_actors().filter("vehicle.*"):
                    if actor.attributes.get("role_name") == "ego_vehicle":
                        self.ego_vehicle = actor
                        return
                time.sleep(0.5)
        except Exception as e:
            self.get_logger().error(f"Błąd połączenia z CARLA: {e}")

    # Tworzenie i konfiguracja sensora Semantic LiDAR
    def spawn_lidar(self):
        try:
            bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            bp.set_attribute('channels', '64')
            bp.set_attribute('range', '80')
            bp.set_attribute('points_per_second', '500000')
            bp.set_attribute('rotation_frequency', '10')
            transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=LIDAR_MOUNT_HEIGHT))
            sensor = self.world.spawn_actor(bp, transform, attach_to=self.ego_vehicle)
            sensor.listen(self.lidar_callback)
            self.get_logger().info('LiDAR semantic uruchomiony.')
            return sensor
        except Exception as e:
            self.get_logger().error(f"Błąd LiDAR: {e}")
            return None

    # Funkcja callback odbierająca dane z LiDAR i przetwarzająca oznakowanie poziome
    def lidar_callback(self, data):
        points = []
        tag24_points = []

        # Filtrowanie punktów z sensora i wyciąganie tych z oznakowaniem poziomym
        for detection in data:
            z_rel = detection.point.z
            if MIN_Z_REL <= z_rel <= MAX_Z_REL:
                point = [detection.point.x, detection.point.y, detection.point.z, detection.cos_inc_angle, detection.object_tag]
                points.append(point)
                if detection.object_tag == TAG_CUSTOM_HORIZONTAL_MARKING:
                    tag24_points.append(point)

        cluster_labels = {}
        class_counts = {k: 0 for k in COLOR_MAP.keys()}

        # Grupowanie punktów oznakowania poziomego i klasyfikacja
        if tag24_points:
            coords = np.array([[p[0], p[1]] for p in tag24_points])
            db = DBSCAN(eps=0.8, min_samples=6).fit(coords)
            labels = db.labels_
            for label_id in set(labels):
                if label_id == -1:
                    continue
                cluster = [tag24_points[i] for i in range(len(labels)) if labels[i] == label_id]
                cluster_xy = np.array([coords[i] for i in range(len(labels)) if labels[i] == label_id])
                mark_type = classify_cluster(cluster_xy)
                class_counts[mark_type] += 1
                for p in cluster:
                    cluster_labels[(p[0], p[1], p[2])] = mark_type

        # Logowanie liczby wykrytych klastrów
        for k, v in class_counts.items():
            if v > 0:
                self.get_logger().info(f"Wykryto {v} klastrów typu: {k}")

        # Tworzenie wiadomości PointCloud2 z kolorami
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "semantic_lidar_link"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="tag", offset=16, datatype=PointField.UINT32, count=1),
            PointField(name="rgb", offset=20, datatype=PointField.UINT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        data_bytes = bytearray()
        for point in points:
            x, y, z, intensity, tag = point
            if tag == TAG_CUSTOM_HORIZONTAL_MARKING:
                mark_type = cluster_labels.get((x, y, z), "unknown")
            else:
                mark_type = "unknown"
            r, g, b = COLOR_MAP.get(mark_type, (1.0, 0.0, 1.0))
            rgb = pack_rgb(r, g, b)
            data_bytes += struct.pack("ffffII", x, y, z, intensity, int(tag), int(rgb))
        msg.data = bytes(data_bytes)
        self.publisher.publish(msg)

    # Publikowanie transformacji sensora w ramce mapy
    def broadcast_tf(self):
        if not self.sensor or not self.ego_vehicle:
            return
        try:
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = "map"
            tf.child_frame_id = "semantic_lidar_link"
            trans = self.sensor.get_transform()
            tf.transform.translation.x = trans.location.x
            tf.transform.translation.y = -trans.location.y
            tf.transform.translation.z = trans.location.z
            r = math.radians(trans.rotation.roll)
            p = -math.radians(trans.rotation.pitch)
            y = -math.radians(trans.rotation.yaw)
            cy = math.cos(y * 0.5)
            sy = math.sin(y * 0.5)
            cp = math.cos(p * 0.5)
            sp = math.sin(p * 0.5)
            cr = math.cos(r * 0.5)
            sr = math.sin(r * 0.5)
            tf.transform.rotation.w = cr * cp * cy + sr * sp * sy
            tf.transform.rotation.x = sr * cp * cy - cr * sp * sy
            tf.transform.rotation.y = cr * sp * cy + sr * cp * sy
            tf.transform.rotation.z = cr * cp * sy - sr * sp * cy
            self.tf_broadcaster.sendTransform(tf)
        except Exception as e:
            self.get_logger().warn(f"Błąd broadcast_tf: {e}")


# Funkcja główna ROS 2

def main(args=None):
    rclpy.init(args=args)
    node = SemanticLidarShowTags()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.sensor:
            if node.sensor.is_listening:
                node.sensor.stop()
            node.sensor.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

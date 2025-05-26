# Semantic LiDAR Carla + ROS 2

Repozytorium zawiera kod i instrukcję uruchomienia węzła ROS 2 do przetwarzania danych z sensora Semantic LiDAR w środowisku Carla.

---

## Wymagania

- ROS 2 Humble
- Docker + NVIDIA GPU
- Carla 0.9.15 (Docker)
- Carla-Autoware-Bridge (Docker)
- Autoware (rocker)
- Python 3.8+
- Pakiety Python: `numpy`, `sklearn`, `rclpy`, `tf2_ros`

---

## Instrukcja uruchomienia

### 1. Uruchomienie Carla

```bash
docker run --privileged --gpus all --net=host -e DISPLAY=$DISPLAY \
  carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen -prefernvidia -quality-level=Low
```

### 2. Uruchomienie Carla-Autoware-Bridge

```bash
docker run -it -e -RMW_IMPLEMENTATION=rmw_cyclonedds_cpp --network host tumgeka/carla-autoware-bridge:latest
```

następnie:
```bash
ros2 launch carla_autoware_bridge carla_aw_bridge.launch.py town:=Town10HD timeout:=500
```

### 3. Uruchomienie ROS 2

W osobnym terminalu:

```bash
rocker --network=host -e  RMW_IMPLEMENTTION=rmw_cyclonedds_cpp -e LIBGL_ALWAYS_SOFTWARE=1 --x11 --nvidia --volume ~/carla:/home/krzysztof-rzym/carla -- ghcr.io/autowarefoundation/autoware:humble-2024.01-cuda-amd64
```

następnie:
```bash
cd /home/krzysztof-rzym/carla/autoware/

source install/setup.bash
	
ros2 launch autoware_launch e2e_simulator.launch.xml vehicle_model:=carla_t2_vehicle sensor_model:=carla_t2_sensor_kit map_path:=/home/krzysztof-rzym/carla/Town10
```

### 4. Uruchomienie Semantic lidar 
```bash
docker cp ~/carla_lidar_ros2/semantic_lidar.py [nazwa kontenera]:/root/

docker exec -it [nazwa kontenera] bash

pip3 install scikit-learn
source /opt/ros/humble/setup.bash
python3 /root/semantic_lidar.py
```
---

## Efekt działania

- W Carla GUI widoczny sensor semantic LiDAR na pojeździe `ego_vehicle`
- W RViz2: chmura punktów `/semantic_lidar_colored` z kolorem zależnym od klasy (np. strzałka w lewo – zielony)
- Konsola ROS 2: logi informujące o rozpoznaniu znaków poziomych

---

## Rozpoznawane oznakowania poziome

| Typ oznakowania        | Kolor w RViz2     | Warunki rozpoznania (przykładowe)      |
|------------------------|-------------------|----------------------------------------|
| Linia stopu           | Czerwony          | Wąska, długa linia                     |
| Strzałka w lewo       | Zielony           | Pionowa strzałka                       |
| Strzałka w prawo      | Niebieski         | Pozioma strzałka                       |
| Strzałka prosto       | Żółty             | Proporcjonalnie szeroka i niska       |
| Pasy dla pieszych     | Błękitny          | Długi i szeroki prostokąt              |
| Linia ciągła (custom) | Biały             | Długa linia (klaster tagu 24)         |
| Inne (TAG 24)         | Różowy            | Nierozpoznane klastry                  |

---

## Zrzuty ekranu

Zrzuty znajdują się w folderze `screenshots/`.

- `arrow_left.png` – wykryta strzałka w lewo
- `stop_line.png` – wykryta linia STOP
- `pedestrian_crossing.png` – wykryte pasy
- `semantic_lidar_rviz2.png` – podgląd w RViz2

---

## Testowane środowisko

- Ubuntu 22.04
- ROS 2 Humble
- Docker 24+
- NVIDIA RTX GPU
- Carla 0.9.15

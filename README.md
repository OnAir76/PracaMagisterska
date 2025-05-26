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
docker run --rm -it --gpus all \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    carlasim/carla:0.9.15 ./CarlaUE4.sh -opengl
```

### 2. Uruchomienie Carla-Autoware-Bridge

```bash
docker compose -f bridge_compose.yaml up
```

Upewnij się, że wybrana jest mapa `Town10HD` oraz sensor kit `carla_t2_sensor_kit`.

### 3. Uruchomienie ROS 2 i węzła semantic_lidar

W osobnym terminalu:

```bash
rocker --nvidia --x11 osrf/ros:humble-desktop
# wewnątrz kontenera:
cd ~/your_workspace/
colcon build
. install/setup.bash

ros2 run your_package semantic_lidar.py
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

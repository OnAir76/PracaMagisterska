# Semantic LiDAR Carla + ROS 2

Repozytorium zawiera kod i instrukcję uruchomienia węzła ROS 2 do przetwarzania danych z sensora Semantic LiDAR w środowisku Carla.

- odbiera dane z sensora `sensor.lidar.ray_cast_semantic` w Carla 0.9.15,
- filtruje punkty z oznakowaniem poziomym (tag 24),
- grupuje je w klastry i klasyfikuje jako:
  - strzałka w lewo (zielony)
  - strzałka w prawo (niebieski)
  - linia STOP (czerwony)
  - pasy dla pieszych (błękitny)
- wizualizuje kolorową chmurę punktów w RViz2,
- wypisuje do terminala liczbę wykrytych znaków.

---

## Wymagania

- ROS 2 Humble
- Docker + NVIDIA GPU
- Carla 0.9.15 (Docker)
- Carla-Autoware-Bridge (Docker)
- Autoware (rocker)
- Python 3.7+
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
### 5. Konfiguracja Skryptu

Najważniejsze parametry, które można modyfikować, znajdują się na początku pliku semantic_lidar_processor.py:
INTERESTING_TAGS: Zbiór tagów semantycznych do analizy. Obecnie ustawiony na {24} (RoadLine).
DBSCAN_EPS: Maksymalna odległość między punktami w klastrze. Zmniejszenie wartości spowoduje tworzenie mniejszych, gęstszych klastrów.
DBSCAN_MIN_SAMPLES: Minimalna liczba punktów do utworzenia klastra.
TEMPLATE_MATCH_THRESHOLD: Próg pewności (od 0.0 do 1.0), powyżej którego detekcja jest uznawana za prawidłową.
SAVE_DEBUG_IMAGES: Ustaw na True, aby zapisywać obrazy klastrów do folderu zdefiniowanego w DEBUG_SAVE_PATH.

### 6. Znane Problemy i Ograniczenia

Detekcja napisu "STOP" i przejść dla pieszych jest problematyczna. Algorytm klasteryzacji ma trudności z połączeniem nieciągłych elementów (osobne litery, pasy "zebry") w jeden spójny obiekt, co prowadzi do błędnej klasyfikacji.

Wyniki są zależne od parametrów DBSCAN oraz progu TEMPLATE_MATCH_THRESHOLD, które mogą wymagać dostrojenia w zależności od scenariusza i warunków w symulacji.


## Efekt działania

- Kolorowa chmura `/semantic_lidar_colored` widoczna w RViz2
- W terminalu:

```
Wykryto 2 klastrów typu: arrow_left
Wykryto 1 klastrów typu: stop_line
```

---

## Rozpoznawane oznakowania poziome

| Typ oznakowania       | Kolor (RGB) |
|------------------------|-------------|
| Strzałka w lewo       | Zielony     |
| Strzałka w prawo      | Niebieski   |
| Linia STOP            | Czerwony    |
| Pasy dla pieszych     | Błękitny    |
| Nierozpoznane         | Różowy      |

---

## Zrzuty ekranu
- wykryta strzałka w lewo i prawo 
![image](https://github.com/user-attachments/assets/447d938f-536f-458e-b98a-e4786b9dbd62)

- Informacja o wykryciu strzałki w prawo
![image](https://github.com/user-attachments/assets/a3b4850f-e767-40fb-a737-88c783c6d01e)

- Informacja o wykryciu strzałki w lewo
![image](https://github.com/user-attachments/assets/d520844c-85bf-4454-be78-af526ae1f441)

- Wzrzec strzałki w prawo
![arrow_right](https://github.com/user-attachments/assets/e68b5374-f61d-411e-ba58-2c9dc6464ba1)

- Wzorzec strzałki w lewo
![arrow_left](https://github.com/user-attachments/assets/bd47b800-22c7-4b35-bcc8-64025cdbe2a2)

---

## Testowane środowisko

- Ubuntu 22.04
- ROS 2 Humble
- Docker 24+
- NVIDIA RTX GPU
- Carla 0.9.15

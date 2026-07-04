import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon # ИМПОРТИРУЕМ POLYGON

class CarSimulator:
    """Моделирование кинематики автомобиля по модели велосипеда (Bicycle Model)."""

    def __init__(self, x: float, y: float, theta: float, L: float = 2.5, 
                 car_length: float = 4.5, car_width: float = 2.0, rear_overhang: float = 1.0):
        self.x = x
        self.y = y
        self.theta = theta  
        self.L = L          
        
        self.length = car_length
        self.width = car_width
        self.overhang = rear_overhang
        
        self.history_x = [x]
        self.history_y = [y]
        self.crashed = False
    
    def get_corners(self):
        """Вычисление глобальных координат 4-х углов кузова через матрицу поворота."""
        front = self.length - self.overhang
        rear = -self.overhang
        half_w = self.width / 2.0
        
        # Порядок важен для Polygon: последовательный обход вершин кузова
        local_corners = np.array([
            [front, half_w],   # Перед-Лево
            [front, -half_w],  # Перед-Право
            [rear, -half_w],   # Зад-Право
            [rear, half_w]     # Зад-Лево
        ]).T 
        
        rot_matrix = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta),  np.cos(self.theta)]
        ])
        
        global_corners = rot_matrix @ local_corners + np.array([[self.x], [self.y]])
        return global_corners.T 

    def check_collision(self) -> bool:
        """Проверка пересечения габаритов с границами парковочного кармана."""
        corners = self.get_corners()
        for cx, cy in corners:
            if cx <= 2.0:
                if cy > 1.5 or cy < -1.5:  
                    return True
        return False

    def update(self, v: float, phi: float, dt: float = 0.1):
        """Интегрирование уравнений движения методом Эйлера."""
        if self.crashed:
            return 
            
        phi = np.clip(phi, -np.radians(30), np.radians(30))
        
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += (v * np.tan(phi) / self.L) * dt
        
        if self.check_collision():
            self.crashed = True


def run_parking_simulation(start_x: float, start_y: float, start_theta: float,
                           k_y: float = 0.4, k_theta: float = 2.5, k_align: float = 3.5) -> CarSimulator:
    """Пропорциональный регулятор (P-controller)."""
    car = CarSimulator(x=start_x, y=start_y, theta=start_theta)
    dt = 0.1
    target_y = 0.0 
    
    for _ in range(300):  
        if car.crashed:
            print(f"[{start_x}, {start_y}] ДТП! Симуляция остановлена.")
            break
            
        if car.x <= -1.5 and abs(car.theta) < 0.05:
            car.update(v=0.0, phi=0.0, dt=dt)
            car.history_x.append(car.x)
            car.history_y.append(car.y)
            break
            
        v = -1.0  
        
        if car.x > 2.0:
            desired_theta = (car.y - target_y) * k_y
            phi = (car.theta - desired_theta) * k_theta
        else:
            phi = car.theta * k_align
            
        car.update(v=v, phi=phi, dt=dt)
        car.history_x.append(car.x)
        car.history_y.append(car.y)
        
    return car


# --- БЛОК ВАЛИДАЦИИ И ТЕСТИРОВАНИЯ ---

test_cases = [
    {"start": (5.0, 1.2, 0.0), "color": "blue", "label": "Тест 1: Близкий старт"},
    {"start": (7.0, 2.0, 0.0), "color": "orange", "label": "Тест 2: Стандартный старт"},
    {"start": (9.0, 3.0, 0.0), "color": "purple", "label": "Тест 3: Дальний старт"},
]

plt.figure(figsize=(12, 7))
plt.axis('equal')  

for case in test_cases:
    x_init, y_init, theta_init = case["start"]
    simulated_car = run_parking_simulation(x_init, y_init, theta_init)
    
    # Визуализация трека
    line_style = '-x' if simulated_car.crashed else '-o'
    plt.plot(simulated_car.history_x, simulated_car.history_y, line_style, 
             markersize=3, color=case["color"], label=case["label"] + (" (ДТП)" if simulated_car.crashed else ""))
    
    # --- НОВАЯ, ИЗЯЩНАЯ ОТРИСОВКА ЧЕРЕЗ POLYGON ---
    car_corners = simulated_car.get_corners()
    poly = Polygon(
        car_corners, 
        closed=True,
        edgecolor='red' if simulated_car.crashed else case["color"], 
        facecolor='red' if simulated_car.crashed else case["color"], 
        alpha=0.25 if not simulated_car.crashed else 0.5
    )
    plt.gca().add_patch(poly)

# Отрисовка карты
plt.plot([-5, 2], [1.1, 1.1], 'k--', linewidth=2, label='Парковочный карман')
plt.plot([-5, 2], [-1.1, -1.1], 'k--', linewidth=2)
plt.plot([2, 2], [-1.1, 1.1], 'k--', linewidth=2)
plt.plot([2, 10], [0, 0], 'g-', alpha=0.3, label='Линия бордюра (Y=0)')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Линия финиша задней оси')

plt.grid(True, linestyle=':', alpha=0.6)
plt.title("Автономная парковка (Kinematic Bicycle Model + P-Controller)")
plt.xlabel("Ось X (метры)")
plt.ylabel("Ось Y (метры)")
plt.legend(loc='upper left')

plt.show()
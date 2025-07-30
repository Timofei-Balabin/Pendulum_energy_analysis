import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# Параметры системы (g/l = 1 в безразмерных единицах)
g_over_l = 1.0  # Так как t = τ*sqrt(g/l), то g/l = 1

tau_max = 35.0  

def pendulum(t, y):
    return [y[1], -np.sin(y[0])]  # Уравнение маятника: d²φ/dt² + sin(φ) = 0

# Начальные условия (разные энергии)
conditions = [
    {'v0': 1.5, 'color': 'blue', 'label': 'E < 2 (колебания)'},
    {'v0': 1.9999, 'color': 'red', 'label': 'E =~ 2 (сепаратриса)'},
    {'v0': 2.1, 'color': 'green', 'label': 'E > 2 (вращение)'}
]

# Решаем ОДУ для каждого случая
solutions = []
for condition in conditions:
    solution = solve_ivp(pendulum, [0, tau_max], [0, condition['v0']], 
                         t_eval=np.linspace(0, tau_max, 1000), rtol=1e-8, atol=1e-8)
    solutions.append(solution)

# Графики угла и фазового портрета
plt.figure(figsize=(8, 5))
for sol, condition in zip(solutions, conditions):
    plt.plot(sol.t, sol.y[0], color=condition['color'], label=condition['label'])
plt.title('Угол от времени')
plt.xlabel('Безразмерное время t, (с)')
plt.ylabel('Угол, (рад)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
for solution, condition in zip(solutions, conditions):
    plt.plot(solution.y[0], solution.y[1], color=condition['color'], label=condition['label'])
plt.title('Фазовый портрет')
plt.xlabel('Угол φ, (рад)')
plt.ylabel('Скорость dφ/dt, (рад/с)')
plt.grid()
plt.legend()
plt.show()

### Расчёт периода T(E) ###
def calculate_period(v0):
    """Численно находит период колебаний для заданной начальной скорости v0."""
    E = v0**2 / 2 - np.cos(0)  # Энергия: E = (dφ/dt)²/2 - cos(φ)
    
    if E < 1.0:  # Малые колебания (E < 1)
        # Период можно найти как время между двумя последовательными прохождениями φ=0
        sol = solve_ivp(pendulum, [0, 50], [0, v0], rtol=1e-8, atol=1e-8)
        crossings = np.where(np.diff(np.sign(sol.y[0])))[0]
        if len(crossings) >= 2:
            return sol.t[crossings[2]] - sol.t[crossings[0]]
    
    elif E < 2.0:  # Колебания вблизи сепаратрисы (1 < E < 2)
        # Находим время от φ=0 до φ=π и умножаем на 2
        def event(t, y):
            return y[0] - np.pi
        event.terminal = True
        sol = solve_ivp(pendulum, [0, 50], [0, v0], events=event, rtol=1e-8, atol=1e-8)
        if sol.t_events[0].size > 0:
            return 2 * sol.t_events[0][0]
    
    else:  # Вращение (E > 2)
        # Период - время, за которое φ меняется на 2π
        def event(t, y):
            return y[0] - 2*np.pi
        event.terminal = True
        sol = solve_ivp(pendulum, [0, 50], [0, v0], events=event, rtol=1e-8, atol=1e-8)
        if sol.t_events[0].size > 0:
            return sol.t_events[0][0]
    
    return np.nan  # Если не удалось найти период

# Аналитические приближения для T(E)
def T_small_oscillations(E):
    """Малые колебания (E << 2)."""
    return 2 * np.pi * (1 + E / 8)  # E в безразмерных единицах (2mgl = 2)

def T_near_separatrix_oscillations(E):
    """Колебания вблизи сепаратрисы (E чуть меньше 2)."""
    return 2 * np.log(32 / (2 - E))

def T_near_separatrix_rotation(E):
    """Вращение вблизи сепаратрисы (E чуть больше 2)."""
    return np.log(32 / (E - 2))

def T_fast_rotation(E):
    """Быстрое вращение (E >> 2)."""
    return np.pi * np.sqrt(2 / E)

# Строим T(E) для разных энергий
E_vals = np.linspace(0.01, 5, 100)
T_num = []  # Численные значения
T_analytical = []  # Аналитические приближения

for E in E_vals:
    if E < 2:
        v0 = np.sqrt(2 * (E + 1))  # E = v0²/2 - cos(0) = v0²/2 - 1
    else:
        v0 = np.sqrt(2 * (E + 1))  # Для E > 2 тоже верно
    
    T_num.append(calculate_period(v0))
    
    # Выбираем аналитическое приближение
    if E < 0.5:
        T_analytical.append(T_small_oscillations(E))
    elif E < 1.99:
        T_analytical.append(T_near_separatrix_oscillations(E))
    elif E < 2.01:
        T_analytical.append(np.inf)  # На сепаратрисе T → ∞
    elif E < 2.5:
        T_analytical.append(T_near_separatrix_rotation(E))
    else:
        T_analytical.append(T_fast_rotation(E))

# График T(E)
plt.figure(figsize=(10, 6))
plt.plot(E_vals, T_num, 'b-', label='Численное решение')
plt.plot(E_vals, T_analytical, 'r--', label='Аналитические приближения')
plt.axvline(x=2, color='k', linestyle=':', label='E = 2 (сепаратриса)')
plt.title('Зависимость периода от энергии T(E)')
plt.xlabel('Энергия E')
plt.ylabel('Период T')
plt.yscale('log')  # Логарифмическая шкала для y
plt.legend()
plt.grid()
plt.show()

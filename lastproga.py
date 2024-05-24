import numpy as np
from scipy.stats import chi2

# Данные
X = [127.3, 124.8, 125.2, 113.8, 124.4, 114.2, 123.7, 126.9, 124.2, 120, 127.3, 119.4, 123, 115.2, 124.4, 129.7, 124.6, 110.2, 120.5, 121.2, 114.8, 121.9, 117, 125.5, 117.7, 117.6, 123.7, 118.7, 124.1, 125.7, 123.6, 124.7, 119.6, 129.2, 129.7, 126.1, 111.4, 118, 117.2, 117.6, 119.8, 119.7, 114.7, 120.6, 122.2, 118.5, 121.6, 120.8, 120.5, 120.8, 122.9, 119.1, 124.1, 124.4, 119.8, 115.1, 124.8, 122.9, 123, 123.7, 126.1, 125.9, 125.2, 122.9, 123.3, 123.8, 121.2, 120.7, 119.3, 129.7, 122.6, 122, 119.4, 130.1, 121.8, 120.4, 113.8, 118.7, 124.1, 123.8, 122.5, 124.4, 115.8, 121.4, 124.4, 119.1, 123.6, 117.5, 118.3, 126.4, 121.8, 124.2, 118.9, 123.9, 126.8, 126.8, 122.6, 118.2, 128, 124, 119.8, 122.4, 116, 118]
Y = [87, 85.5, 79.8, 76.4, 82.7, 79.6, 78.4, 87.9, 77.7, 79.3, 87.5, 77.7, 81.5, 78.6, 84.2, 82.7, 83.1, 75.5, 83.6, 81.3, 78, 75.9, 83.1, 84, 83.3, 76.6, 84.1, 75.7, 82.7, 86.4, 78.7, 78.5, 83.9, 86.2, 83.1, 82.3, 70.4, 82.5, 80.2, 73.6, 82.1, 79.9, 76.7, 78.1, 82, 77.3, 82.4, 79.5, 79, 82.7, 84.8, 77.9, 80.7, 83.3, 82.1, 80.4, 83.6, 79.4, 80.4, 83, 87.7, 78.8, 81.8, 83.8, 81.9, 82.6, 77.1, 78.6, 78.3, 80.4, 79.6, 76.8, 78.6, 81.4, 80.7, 77.4, 74.4, 78.2, 80.4, 81.1, 87, 82.3, 79.4, 83.2, 79.9, 81.8, 84.1, 74.2, 77.7, 84.2, 79.3, 84, 79.7, 83.2, 83.9, 79.5, 82.4, 79.4, 83.8, 79.5, 80.9, 75.7, 72.1, 77.6]

# Параметры
r = 4  # Количество интервалов для X
s = 5  # Количество интервалов для Y
alpha = 0.05

# Границы интервалов
X_bins = [118.05, 120.05, 122.05, 124.05, 126.05]
Y_bins = [78.05, 79.25, 80.45, 81.65, 82.85, 84.05]

# Подсчет частот попаданий
contingency_table = np.histogram2d(X, Y, bins=[X_bins, Y_bins])[0]

# Рассчет статистики хи-квадрат
n = np.sum(contingency_table)  # Общее количество наблюдений
row_sums = np.sum(contingency_table, axis=1)  # Суммы по строкам
col_sums = np.sum(contingency_table, axis=0)  # Суммы по столбцам


expected = np.outer(row_sums, col_sums) / n  # Ожидаемые частоты
chi2_stat = np.sum((contingency_table - expected) ** 2 / expected)  # Статистика хи-квадрат

# Количество степеней свободы
df = (r - 1) * (s - 1)

# Критическое значение
critical_value = chi2.ppf(1 - alpha, df)

# Вывод результатов
print(f"Таблица сопряженности:\n{contingency_table}")
print(f"Статистика хи-квадрат: {chi2_stat}")
print(f"Критическое значение: {critical_value}")

# Решение о гипотезе
if chi2_stat > critical_value:
    print("Отвергаем нулевую гипотезу: параметры X и Y не являются независимыми.")
else:
    print("Не отвергаем нулевую гипотезу: параметры X и Y независимы.")
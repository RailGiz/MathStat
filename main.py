import pandas as pd
from scipy import stats
import numpy as np
print("ZADANIE1")
# Загрузите данные из Excel
xls = pd.ExcelFile('var4.xls')

# Читаем данные из листов
sheet1 = xls.parse('Sheet1')
sheet2 = xls.parse('Sheet2')

# Получаем данные до и после воздействия
before = sheet2['Z5A']
after = sheet2['Z5B']

# Удаление значений NaN
before = [x for x in before if str(x) != 'nan']
after = [x for x in after if str(x) != 'nan']

# Получаем уровень значимости и направление ожиданий исследователя
alpha = 0.05
expectation = 'increase'

# Вычисляем разности
diff = np.array(before) - np.array(after)

# Вычисляем среднее и стандартное отклонение
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)

# Вычисляем статистику Стьюдента
T = mean_diff / (std_diff / np.sqrt(len(before)))

# Вычисляем p-value
if expectation == 'increase':
    p_value = 1 - stats.t.cdf(T, df=len(before)-1)
else:
    p_value = stats.t.cdf(T, df=len(before)-1)

# Выводим результаты
print("До:")
print(f"Объем выборки: {len(before)}")
print(f"Среднее: {np.mean(before):.2f}")
print(f"Станд. отклонение: {np.std(before, ddof=1):.2f}")
print(f"Станд. ошибка среднего: {np.std(before, ddof=1) / np.sqrt(len(before)):.2f}")

print("\nПосле:")
print(f"Объем выборки: {len(after)}")
print(f"Среднее: {np.mean(after):.2f}")
print(f"Станд. отклонение: {np.std(after, ddof=1):.2f}")
print(f"Станд. ошибка среднего: {np.std(after, ddof=1) / np.sqrt(len(after)):.2f}")

print("\nПо разностям:")
print(f"Объем выборки: {len(diff)}")
print(f"Среднее: {mean_diff:.2f}")
print(f"Станд. отклонение: {std_diff:.2f}")
print(f"Станд. ошибка среднего: {std_diff / np.sqrt(len(diff)):.2f}")
print(f"\n5%-ая критическая область: T >= {stats.t.ppf(1 - alpha, df=len(before)-1):.2f}")
print(f"\nC_Alpha: = {stats.t.ppf(1 - alpha, df=len(before)-1):.2f}")
print(f"Статистика Стьюдента: {T}")
print(f"С критическим уровнем значимости 𝑝𝑣𝑎𝑙𝑢𝑒 = {p_value}")

if p_value < alpha:
    print("Вывод: Нулевая гипотеза отвергается")
else:
    print("Вывод: Нулевая гипотеза принимается")


import pandas as pd
from scipy import stats
import numpy as np
print("ZADANIE2")
# Загрузите данные из Excel
xls = pd.ExcelFile('var4.xls')

# Читаем данные из листа 'Sheet2'
sheet2 = xls.parse('Sheet2')

# Получаем данные для двух приборов
device1 = sheet2['Z9A']
device2 = sheet2['Z9B']

# Удаление значений NaN
device1 = [x for x in device1 if str(x) != 'nan']
device2 = [x for x in device2 if str(x) != 'nan']

# Получаем уровень значимости
alpha = 0.025  # Замените на ваше значение alpha

# Вычисляем дисперсии
var1 = np.var(device1, ddof=1)
var2 = np.var(device2, ddof=1)

# Проводим два теста Фишера
for expectation in ['less', 'greater']:
    # Вычисляем статистику F
    F = var1 / var2 if expectation == 'greater' else var2 / var1

    # Вычисляем p-value
    df1 = len(device1) - 1
    df2 = len(device2) - 1
    p_value = 1 - stats.f.cdf(F, df1, df2) if expectation == 'greater' else stats.f.cdf(F, df1, df2)

    # Вычисляем критическую константу C_alpha
    C_alpha = stats.f.ppf(1 - alpha, df1, df2) if expectation == 'greater' else stats.f.ppf(alpha, df1, df2)

    # Выводим результаты для каждого прибора
    for i, device in enumerate([device1, device2], start=1):
        print(f"\nПрибор {i}:")
        print(f"Объем выборки: {len(device)}")
        print(f"Среднее: {np.mean(device):.2f}")
        print(f"Несмещенная оценка дисперсии: {np.var(device, ddof=1):.2f}")

    print(f"\nОжидание: {expectation}")
    print(f"Статистика F: {F:.2f}")
    print(f"p-value: {p_value:.3f}")
    print(f"Критическая константа C_alpha: {C_alpha:.2f}")

    if p_value < alpha:
        print("Вывод: Нулевая гипотеза отвергается")
    else:
        print("Вывод: Нулевая гипотеза принимается")

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
print("ZADANIE3")
# Загрузите данные из Excel
xls = pd.ExcelFile('var4.xls')

# Читаем данные из листа 'Sheet2'
sheet2 = xls.parse('Sheet2')

# Получаем данные для события A
event_A = sheet2['Z6I']

# Удаление значений NaN
event_A = [x for x in event_A if str(x) != 'nan']

# Подсчет частоты появления A
freq_A = event_A.count('A')
total_events = len(event_A)
p_A = freq_A / total_events

print(f"Частота появления A: {freq_A}")

# Проверка гипотезы
# H0: p = 0.5
# H1: p < 0.5
alpha = 0.05
z, pvalue = proportions_ztest(freq_A, total_events, 0.2, alternative='smaller')

if pvalue < alpha:
    print("Нулевая гипотеза отвергается, pvalue =", pvalue)
else:
    print("Нулевая гипотеза принимается, pvalue =", pvalue)


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
print("ZADANIE4")
# Загрузка данных из файла Excel
xls_file = pd.ExcelFile('var4.xls')
df = xls_file.parse('Sheet2')
alpha = 0.025
# Создание двух групп
group1 = df['Z8A'].dropna().values
group2 = df['Z8B'].dropna().values

# Выполнение теста Вилкоксона
statistic, p_value = stats.ranksums(group1, group2)

# Определение направления ожиданий исследователя
if p_value < alpha:
    direction = "больше"
else:
    direction = "меньше"

# Дополнительные результаты
sample_size_group1 = len(group1)
sample_size_group2 = len(group2)
W_sum = np.sum(stats.rankdata(np.concatenate((group1, group2))))
mu_W = sample_size_group1 * (sample_size_group1 + sample_size_group2 + 1) / 2
sigma_W = np.sqrt(sample_size_group1 * sample_size_group2 * (sample_size_group1 + sample_size_group2 + 1) / 12)

# Вывод результатов
print(f"Уровень значимости (альфа): {alpha:.4f}")
print(f"Объем выборки (group1): {sample_size_group1}")
print(f"Объем выборки (group2): {sample_size_group2}")
print(f"Сумма рангов (W): {W_sum:.4f}")
print(f"Математическое ожидание ранга (𝜇W): {mu_W:.4f}")
print(f"Стандартное отклонение рангов (𝜎W): {sigma_W:.4f}")
print(f"Статистика теста Вилкоксона: {abs(statistic):.4f}")  # Исправлено: берем абсолютное значение
print(f"p-значение: {p_value:.4f}")
print(f"Ожидание исследователя: Средний ранг {direction} между двумя группами.")
if p_value < alpha:
    print("Нулевая гипотеза отвергается")
else:
    print("Нулевая гипотеза не отвергается")
# Создание графиков линий для эмпирических функций распределения
plt.figure(figsize=(8, 6))
plt.plot(np.sort(group1), np.arange(1, len(group1) + 1) / len(group1), label='Group 1', color='blue')
plt.plot(np.sort(group2), np.arange(1, len(group2) + 1) / len(group2), label='Group 2', color='orange')
plt.xlabel('Value')
plt.ylabel('Кумулятивная вероятность')
plt.title('Сравнение эмпирических функций распределения')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
from scipy.stats import chi2
print("ZADANIE5")
# Здесь вставьте название вашего файла Excel
file_name = "var4.xls"

# Здесь вставьте названия переменных и столбцов
alpha = 0.025  # Уровень значимости
delta = 1.0  # Погрешность
x0 = 18.8  # Начальное приближение

group1_column = "Z10A"
group2_column = "Z10B"

# Чтение данных из Excel
data = pd.read_excel(file_name, sheet_name=1)  # Предполагается, что данные находятся на второй странице

# Выделение выборок из двух групп
group1 = data[group1_column].values
group2 = data[group2_column].values


# Функция для вычисления статистики критерия хи-квадрат
def chi_square_statistic(group1, group2):
    n1 = len(group1)
    n2 = len(group2)
    r = min(len(set(group1)), len(set(group2)))  # Количество интервалов

    observed_freq_group1 = pd.Series(group1).value_counts(sort=False)
    observed_freq_group2 = pd.Series(group2).value_counts(sort=False)

    chi_square = 0
    for j in range(r):
        nu_j1 = observed_freq_group1.get(j, 0)
        nu_j2 = observed_freq_group2.get(j, 0)
        nu_j_star = nu_j1 + nu_j2

        if nu_j_star != 0:  # Проверка на ноль
            chi_square += ((nu_j1 / n1) - (nu_j2 / n2)) ** 2 / nu_j_star

    chi_square *= n1 * n2
    return chi_square


# Вычисление статистики критерия
X2 = chi_square_statistic(group1, group2)

# Нахождение критической константы
df = min(len(set(group1)), len(set(group2))) - 1
critical_value = chi2.ppf(1 - alpha, df)

# Вывод результатов
print("Статистика критерия Хи-квадрат:", X2)
print("Критическая константа:", critical_value)

# Проверка гипотезы
if X2 > critical_value:
    print("Отвергаем нулевую гипотезу: данные не однородны.")
else:
    print("Принимаем нулевую гипотезу: данные однородны.")

# Вычисление p-value
p_value = 1 - chi2.cdf(X2, df)
print("p-value:", p_value)

# Вывод результата на основе p-value
if p_value < alpha:
    print("Отвергаем нулевую гипотезу: данные не однородны.")
else:
    print("Принимаем нулевую гипотезу: данные однородны.")

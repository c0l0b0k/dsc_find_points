import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks

from comon_func_find_events import load_netzsch5, find_events

# Пути к папкам
LABELS_PATH = "data/labels"
DATA_PATH = "data/data_piroliz_10"

# Названия столбцов
TEMP_COL = "Temp./°C"
DSC_COL = "DSC/(uV/mg)"

# Флаг: использовать ли разметки
USE_LABELS = False

# Загрузка разметок (если нужно)
grouped = None
if USE_LABELS and os.path.exists(LABELS_PATH):
    all_labels = pd.concat([
        pd.read_csv(os.path.join(LABELS_PATH, fname))
        for fname in os.listdir(LABELS_PATH) if fname.endswith(".csv")
    ])
    grouped = all_labels.groupby("filename")

# Список файлов
data_files = sorted(os.listdir(DATA_PATH))
if USE_LABELS:
    data_files = list(grouped.groups.keys())

# Только первые 5
selected_files = data_files[:5]

# Сетка 2 строки × 3 столбца
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, filename in enumerate(selected_files):
    data_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(data_path):
        print(f"[!] Файл данных не найден: {filename}")
        continue

    try:
        df = load_netzsch5(data_path)
    except Exception as e:
        print(f"[!] Ошибка чтения файла {filename}: {e}")
        continue

    df.columns = [col.strip() for col in df.columns]

    if TEMP_COL not in df.columns or DSC_COL not in df.columns:
        print(f"[!] Нет нужных столбцов в файле: {filename}")
        continue

    temp = df[TEMP_COL].astype(float)
    dsc = df[DSC_COL].astype(float)

    ax = axes[idx]
    ax.plot(temp, dsc, label="DSC", color="black")

    if USE_LABELS and grouped is not None and filename in grouped.groups:
        events = grouped.get_group(filename)
        for _, row in events.iterrows():
            try:
                start_temp = float(row["temp_start_idx"])
                end_temp = float(row["temp_end_idx"])
                label = row["label"]

                idx1 = (temp - start_temp).abs().idxmin()
                idx2 = (temp - end_temp).abs().idxmin()

                if idx1 < 0 or idx2 >= len(temp):
                    continue

                ax.axvspan(temp.iloc[idx1], temp.iloc[idx2], alpha=0.3, label=label)

            except Exception as e:
                print(f"[!] Ошибка в {filename}: {e}")
                continue

    ax.set_title(f"{filename}", fontsize=10)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("DSC (uV/mg)")
    ax.legend(fontsize=8)
    ax.grid()

    # Автоматическое определение событий
    find_events(filename, df, fig, ax)

# Удалить лишние графики
if len(selected_files) < 6:
    for i in range(len(selected_files), 6):
        fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

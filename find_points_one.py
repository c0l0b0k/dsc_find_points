import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks

from comon_func_find_events import load_netzsch5, find_events

# Пути к папкам
LABELS_PATH = "data/labels"
DATA_PATH = "data/data_piroliz_20"

# Названия столбцов
TEMP_COL = "Temp./°C"
DSC_COL = "DSC/(uV/mg)"

# Флаг: использовать ли разметки
USE_LABELS = False

# Загружаем разметки, если включено
grouped = {}
if USE_LABELS and os.path.exists(LABELS_PATH):
    all_labels = pd.concat([
        pd.read_csv(os.path.join(LABELS_PATH, fname))
        for fname in os.listdir(LABELS_PATH) if fname.endswith(".csv")
    ])
    grouped = all_labels.groupby("filename")
    file_list = grouped.groups.keys()
else:
    # Просто берём все файлы из DATA_PATH
    file_list = [fname for fname in os.listdir(DATA_PATH) if fname.endswith(".txt")]

# Проходим по каждому файлу
for filename in file_list:
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

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(temp, dsc, label="DSC", color="black")

    # Рисуем разметки, если они есть
    if USE_LABELS and filename in grouped:
        for _, row in grouped.get_group(filename).iterrows():
            try:
                start_temp = float(row["temp_start_idx"])
                end_temp = float(row["temp_end_idx"])
                label = row["label"]

                idx1 = (temp - start_temp).abs().idxmin()
                idx2 = (temp - end_temp).abs().idxmin()

                if idx1 < 0 or idx2 >= len(temp):
                    print(f"[!] Пропуск области в {filename}: индексы вне диапазона ({idx1}, {idx2}) при длине {len(temp)})")
                    continue

                ax1.axvspan(temp.iloc[idx1], temp.iloc[idx2], alpha=0.3, label=label)

            except Exception as e:
                print(f"[!] Ошибка обработки области в {filename}: {e}")
                continue

    ax1.set_title(f"DSC: {filename}")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("DSC (uV/mg)")
    ax1.legend()
    fig.tight_layout()

    # Найти события и дополнительно визуализировать
    plt = find_events(filename, df, fig, ax1)
    plt.show()

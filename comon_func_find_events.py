import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks



# Пути к папкам с разметками и данными
LABELS_PATH = "data/labels"
DATA_PATH = "data/data_piroliz_10"

# Названия столбцов в данных
TEMP_COL = "Temp./°C"
DSC_COL = "DSC/(uV/mg)"

def load_netzsch5(path, enc="cp1251"):
    with open(path, encoding=enc, errors="replace") as f:
        for ln, line in enumerate(f):
            if line.startswith("##"):
                start = ln + 1
                break
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        header=None,
        skiprows=start,
        usecols=[0, 2],          # 0‑ Temp, 2‑ DSC
        names=[TEMP_COL, DSC_COL],
        encoding=enc,
        engine="python"
    )


from typing import List, Tuple

def detect_event_points_only(
    temp: pd.Series,
    dsc: pd.Series,
    window=6,
    poly=3,
    min_temp=180,
    threshold=0.001,
    exclusion_range=30
):
    if window and poly:
        dsc_smooth = savgol_filter(dsc, window, poly)
        d1 = savgol_filter(dsc, window, poly, deriv=1)
        d2 = savgol_filter(dsc, window, poly, deriv=2)
    else:
        dsc_smooth = dsc.copy()
        d1 = np.gradient(dsc, temp)
        d2 = np.gradient(d1, temp)

    points = []

    # 1. Основные точки — пересечения нуля второй производной
    for i in range(1, len(d2)):
        if temp.iloc[i] < min_temp:
            continue
        if d2[i - 1] * d2[i] < 0:
            points.append(i)

    # 2. Дополнительные точки
    for i in range(1, len(d2) - 1):
        t = temp.iloc[i]
        if t < min_temp or abs(d2[i]) >= threshold:
            continue
        if any(abs(t - temp.iloc[j]) < exclusion_range for j in points):
            continue
        points.append(i)

    # 3. Закрытие последнего пика, если нужно
    peak_idx = dsc[temp > min_temp].idxmax()
    has_right = any(i > peak_idx for i in points)

    if not has_right:
        right_part = d2[peak_idx + 1:]
        if right_part.size > 0:
            max_d2_idx = peak_idx + 1 + np.argmax(right_part)
            points.append(max_d2_idx)

    # Упорядочим по температуре
    points = sorted(points, key=lambda i: temp.iloc[i])

    return points, dsc_smooth, d1, d2

def find_convex_segments(temp: pd.Series, dsc: np.ndarray, points: list[int], max_temp_span=450):
    """
    Возвращает отрезки, соединяющие пары точек, между которыми весь DSC выше прямой,
    и разность температур не превышает max_temp_span.
    """
    segments = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            idx1, idx2 = points[i], points[j]
            t1, t2 = temp.iloc[idx1], temp.iloc[idx2]

            # Ограничение по температуре
            if abs(t2 - t1) > max_temp_span:
                continue

            y1, y2 = dsc[idx1], dsc[idx2]
            slope = (y2 - y1) / (t2 - t1)
            intercept = y1 - slope * t1

            valid = True
            for k in range(idx1 + 1, idx2):
                y_line = slope * temp.iloc[k] + intercept
                if dsc[k] < y_line:
                    valid = False
                    break

            if valid:
                segments.append((idx1, idx2))

    return segments

from collections import defaultdict

def group_segments_by_overlap(segments, temp, overlap_eps=10):
    """
    Группирует отрезки (idx1, idx2) по пересечению по температуре.
    Возвращает список групп: каждая группа — список отрезков.
    """
    # 1. Создаём список температурных интервалов
    intervals = [
        (i, min(temp.iloc[s[0]], temp.iloc[s[1]]), max(temp.iloc[s[0]], temp.iloc[s[1]]))
        for i, s in enumerate(segments)
    ]

    # 2. Строим граф по пересечению интервалов
    graph = defaultdict(set)
    for i, start1, end1 in intervals:
        for j, start2, end2 in intervals:
            if i == j:
                continue
            # проверка перекрытия с допуском
            if min(end1, end2) - max(start1, start2) >= -overlap_eps:
                graph[i].add(j)
                graph[j].add(i)

    # 3. Поиск компонент связности (классика)
    visited = set()
    groups = []

    def dfs(node, group):
        visited.add(node)
        group.append(segments[node])
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for i in range(len(segments)):
        if i not in visited:
            group = []
            dfs(i, group)
            groups.append(group)

    return groups


# не всегда работает верно из-за костыля по ограничению длины в 450 для событий
# def select_representative_segments(groups, temp):
#     """
#     Выбирает по одной (главной) линии из каждой группы — самую длинную по температуре.
#     """
#     result = []
#     for group in groups:
#         best = max(group, key=lambda s: abs(temp.iloc[s[1]] - temp.iloc[s[0]]))
#         result.append(best)
#     return result

def build_main_lines_for_groups(groups: list[list[tuple[int, int]]], temp: pd.Series, dsc: np.ndarray):
    """
    Для каждой группы строит "главную линию" — отрезок между двумя точками с максимальной температурной разницей.
    Возвращает список пар индексов (idx1, idx2).
    """
    main_lines = []

    for group in groups:
        point_set = set()
        for idx1, idx2 in group:
            point_set.add(idx1)
            point_set.add(idx2)
        point_list = sorted(point_set, key=lambda i: temp.iloc[i])

        if len(point_list) >= 2:
            idx1 = point_list[0]
            idx2 = point_list[-1]
            main_lines.append((idx1, idx2))

    return main_lines

def find_events(path, df, fig, ax1):
    temp = df[TEMP_COL].astype(float)
    dsc = df[DSC_COL].astype(float)

    points, dsc_smooth, d1, d2 = detect_event_points_only(temp, dsc)

    d1_norm = d1 / np.max(np.abs(d1)) if np.max(np.abs(d1)) != 0 else d1
    d2_norm = d2 / np.max(np.abs(d2)) if np.max(np.abs(d2)) != 0 else d2

    ax1.plot(temp, dsc_smooth, label="DSC (smooth)", linewidth=2, color='blue')

    for i, idx in enumerate(points):
        ax1.plot(temp.iloc[idx], dsc_smooth[idx], 'ro')
        ax1.text(temp.iloc[idx], dsc_smooth[idx], f"P{i + 1}", fontsize=8)

    segments = find_convex_segments(temp, dsc, points, max_temp_span=400)
    groups = group_segments_by_overlap(segments, temp)

    # Отрисовать разными цветами каждую группу
    colors = ['yellow', 'magenta', 'lime', 'cyan', 'orange']
    for gi, group in enumerate(groups):
        for idx1, idx2 in group:
            ax1.plot(
                [temp.iloc[idx1], temp.iloc[idx2]],
                [dsc[idx1], dsc[idx2]],
                color=colors[gi % len(colors)],
                linewidth=2,
                zorder=2
            )

    # Главные линии
    main_lines = build_main_lines_for_groups(groups, temp, dsc)



    # Отрисовка главных линий поверх
    for idx1, idx2 in main_lines:
        ax1.plot([temp.iloc[idx1], temp.iloc[idx2]], [dsc[idx1], dsc[idx2]], color='black', linewidth=3, zorder=3)

    ax1.legend(loc="upper left")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(temp, d1_norm,  color='red', alpha=0.7)#label="1st derivative (norm)",
    ax2.plot(temp, d2_norm,  color='green', linestyle="--", alpha=0.7)#label="2nd derivative (norm)",
    ax2.set_ylabel("Normalized derivatives", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.axhline(0, color='black', linestyle=':', linewidth=1)
    ax2.legend(loc="upper right")

    fig.suptitle("DSC: ключевые точки", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    return plt

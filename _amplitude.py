import librosa
import numpy as np
import os
from datetime import timedelta


def is_amplitude_constant(y, sr, window_size=1.0, max_rms_change=0.001, min_duration=500.0):
    """
    Проверка, есть ли участки, где амплитуда почти не меняется дольше min_duration секунд.
    """
    n_samples_per_window = int(window_size * sr)
    rms_values = []
    constant_segments = []
    current_segment_start = None

    for start in range(0, len(y), n_samples_per_window):
        end = start + n_samples_per_window
        if end > len(y):
            break

        segment = y[start:end]
        rms = np.sqrt(np.mean(segment ** 2))
        rms_values.append(rms)
        time_sec = start / sr

        if len(rms_values) > 1:
            rms_change = abs(rms_values[-1] - rms_values[-2])

            if rms_change <= max_rms_change:
                if current_segment_start is None:
                    current_segment_start = time_sec - window_size
            else:
                if current_segment_start is not None:
                    segment_duration = time_sec - current_segment_start
                    if segment_duration >= min_duration:
                        constant_segments.append((
                            current_segment_start,
                            time_sec,
                            segment_duration
                        ))
                    current_segment_start = None

    if current_segment_start is not None:
        segment_duration = (len(y) / sr) - current_segment_start
        if segment_duration >= min_duration:
            constant_segments.append((
                current_segment_start,
                len(y) / sr,
                segment_duration
            ))

    return len(constant_segments) > 0, constant_segments


def scan_directory_for_bad_files(directory, **kwargs):
    """
    Сканирует директорию и находит файлы с "застывшей" амплитудой > 60 сек.
    """
    if not os.path.isdir(directory):  # Проверяем, что это папка, а не файл
        raise ValueError(f"Указанный путь не является папкой: {directory}")

    bad_files = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(".mp3"):  # Проверяем расширение без учета регистра
            file_path = os.path.join(directory, filename)
            try:
                y, sr = librosa.load(file_path, sr=None, mono=True)
                has_issue, segments = is_amplitude_constant(y, sr, **kwargs)
                if has_issue:
                    bad_files[filename] = segments
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")

    return bad_files


def format_time(seconds):
    """Конвертирует секунды в формат MM:SS"""
    return str(timedelta(seconds=seconds))[2:7]


# !
if __name__ == "__main__":
    directory = r"C:\temp\!"  #  путь к папке
    try:
        bad_files = scan_directory_for_bad_files(
            directory,
            window_size=1.0,
            max_rms_change=0.001,
            min_duration=60.0
        )

        if not bad_files:
            print("Файлы с проблемными участками не найдены.")
        else:
            print("Найдены файлы с неизменной амплитудой > 60 сек:")
            for filename, segments in bad_files.items():
                print(f"\nФайл: {filename}")
                for start, end, duration in segments:
                    print(
                        f"  — Начало: {format_time(start)}, "
                        f"Конец: {format_time(end)}, "
                        f"Длительность: {duration:.1f} сек"
                    )
    except Exception as e:
        print(f"Ошибка: {e}")
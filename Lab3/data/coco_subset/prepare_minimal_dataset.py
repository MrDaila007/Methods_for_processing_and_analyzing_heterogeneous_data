#!/usr/bin/env python3
"""
Скрипт для создания минимального тестового датасета из изображений demo.
Создает простые аннотации для тестирования функциональности сравнения моделей.

Использование:
    python prepare_minimal_dataset.py
"""

from pathlib import Path
import shutil

# Пути
LAB3_ROOT = Path(__file__).parent.parent.parent
DEMO_DIR = LAB3_ROOT / "data" / "demo"
COCO_SUBSET_DIR = LAB3_ROOT / "data" / "coco_subset"
IMAGES_DIR = COCO_SUBSET_DIR / "images" / "train"
LABELS_DIR = COCO_SUBSET_DIR / "labels" / "train"

# Создаем директории
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Копируем изображения из demo
demo_images = list(DEMO_DIR.glob("*.jpg"))
demo_images = [img for img in demo_images if not img.name.startswith("pred_")]  # исключаем результаты детекции

if not demo_images:
    print("❌ Нет изображений в data/demo/")
    print("   Добавьте изображения в data/demo/ или используйте существующий датасет COCO")
    exit(1)

print(f"📸 Найдено {len(demo_images)} изображений в demo/")

# Копируем изображения и создаем простые аннотации
# Примечание: это создает фиктивные аннотации для демонстрации структуры
# Для реального сравнения нужны правильные аннотации!

copied_count = 0
for img_path in demo_images[:10]:  # максимум 10 изображений
    # Копируем изображение
    dst_img = IMAGES_DIR / img_path.name
    if not dst_img.exists():
        shutil.copy2(img_path, dst_img)
        copied_count += 1
    
    # Создаем пустой файл аннотации (пользователь должен добавить реальные аннотации)
    label_file = LABELS_DIR / (img_path.stem + ".txt")
    if not label_file.exists():
        # Создаем файл с комментарием
        with open(label_file, "w") as f:
            f.write("# Добавьте аннотации в формате YOLO:\n")
            f.write("# class_id x_center y_center width height\n")
            f.write("# Пример: 0 0.5 0.5 0.3 0.4  # person в центре\n")
        print(f"   Создан файл аннотации: {label_file.name}")

print(f"\n✅ Скопировано {copied_count} изображений в {IMAGES_DIR}")
print(f"✅ Создано {copied_count} файлов аннотаций в {LABELS_DIR}")
print("\n⚠️  ВАЖНО: Файлы аннотаций содержат только шаблоны!")
print("   Для реального сравнения моделей добавьте правильные аннотации в формате YOLO.")
print("   Формат: class_id x_center y_center width height (все значения 0-1)")
print("\n   Пример аннотации для изображения с человеком:")
print("   0 0.5 0.5 0.3 0.4  # person в центре изображения")
print("\n📖 Подробнее см. README.md в этой директории")


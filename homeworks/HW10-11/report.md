# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

В части A использован датасет STL10 для задачи классификации изображений.  
Этот датасет содержит реальные изображения объектов и подходит для демонстрации преимуществ CNN и transfer learning.

В части B использован датасет OxfordIIITPet и выбран трек segmentation.  
Датасет содержит изображения кошек и собак с пиксельными масками, что позволяет корректно оценить качество сегментации.

В части A сравнивались четыре модели:

C1 — простая CNN без аугментаций  
C2 — CNN с аугментациями  
C3 — pretrained ResNet18 (обучается только классификационная голова)  
C4 — pretrained ResNet18 с partial fine-tuning  

Во второй части сравнивались два варианта постобработки масок:

V1 — базовая постобработка  
V2 — альтернативная постобработка с другим порогом вероятности  

---

## 2. Среда и воспроизводимость

Python: 3.10  

torch  
torchvision  
numpy  
matplotlib  

Устройство:
cuda при наличии GPU, иначе cpu  

Seed:
42  

Как запустить:
открыть HW10-11.ipynb и выполнить Run All.

Все эксперименты воспроизводимы благодаря фиксированному seed и фиксированному разбиению данных.

---

## 3. Данные

### 3.1 Часть A: классификация

Датасет:
STL10

Разделение данных:
train / val / test  

Валидационная выборка формируется из train с использованием фиксированного seed.

Transforms:

базовые:
Resize  
ToTensor  
Normalize  

аугментации:
RandomHorizontalFlip  
RandomCrop  
ColorJitter  

Комментарий:

STL10 содержит 10 классов изображений объектов реального мира.  
Изображения имеют достаточно сложную структуру и разнообразие, что делает задачу классификации нетривиальной и хорошо показывает преимущества transfer learning.

---

### 3.2 Часть B: structured vision

Датасет:
OxfordIIITPet

Трек:
segmentation

Ground truth:
пиксельные маски животных

Foreground:
кошка или собака

Предсказания:
используется pretrained модель FCN ResNet50 из torchvision.

Комментарий:

датасет содержит изображения домашних животных с разметкой пикселей.  
Это делает задачу сегментации понятной и позволяет использовать метрику IoU.

---

## 4. Часть A: модели и обучение (C1-C4)

C1 (simple-cnn-base):
простая CNN без аугментаций

C2 (simple-cnn-aug):
та же архитектура CNN, но используются аугментации изображений

C3 (resnet18-head-only):
используется pretrained ResNet18, обучается только последний слой

C4 (resnet18-finetune):
используется pretrained ResNet18, дополнительно обучается часть слоев сети

Loss:
CrossEntropyLoss

Optimizer:
Adam

Batch size:
128

Epochs:
10

Критерий выбора лучшей модели:
максимальное значение val_accuracy.

---

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

Модель:
FCN ResNet50 pretrained

Foreground:
объект животного

V1:
бинаризация маски через argmax

V2:
бинаризация маски через порог вероятности

Метрика:
mean IoU

Дополнительные метрики:

pixel precision  
pixel recall  

Метрики рассчитываются по сравнению предсказанной маски с ground truth.

---

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: [`./artifacts/runs.csv`](./artifacts/runs.csv)

- Лучшая модель классификации: [`./artifacts/best_classifier.pt`](./artifacts/best_classifier.pt)

- Конфиг лучшей модели: [`./artifacts/best_classifier_config.json`](./artifacts/best_classifier_config.json)

- Кривые обучения лучшей модели: [`./artifacts/figures/classification_curves_best.png`](./artifacts/figures/classification_curves_best.png)

- Сравнение моделей C1-C4: [`./artifacts/figures/classification_compare.png`](./artifacts/figures/classification_compare.png)

- Визуализация аугментаций: [`./artifacts/figures/augmentations_preview.png`](./artifacts/figures/augmentations_preview.png)

- Примеры сегментации: [`./artifacts/figures/segmentation_examples.png`](./artifacts/figures/segmentation_examples.png)

- Метрики сегментации: [`./artifacts/figures/segmentation_metrics.png`](./artifacts/figures/segmentation_metrics.png)


Короткая сводка:

Лучший эксперимент части A:
C4

Лучшая val_accuracy:
0.946

Test accuracy лучшей модели:
0.944

Аугментации улучшили результат по сравнению с базовой CNN.

Transfer learning значительно увеличил точность классификации.

Fine-tuning показал лучший результат по сравнению с обучением только классификационной головы.

Сегментационная модель корректно выделяет объект.

V1 показывает более стабильную форму маски.

V2 даёт немного более шумные границы.

Mean IoU показывает хорошее качество сегментации.

---

## 7. Анализ

Простая CNN показывает более низкую точность, так как обучается с нуля и не использует предварительно извлечённые признаки.

Добавление аугментаций увеличивает устойчивость модели к различным вариантам изображений и повышает точность классификации.

Использование pretrained ResNet18 значительно улучшает качество классификации, так как сеть уже обучена извлекать важные признаки изображений.

Head-only обучение даёт хороший результат, однако partial fine-tuning позволяет адаптировать признаки к конкретному датасету.

Метрика IoU подходит для задачи сегментации, так как показывает степень совпадения предсказанной маски с правильной.

При изменении способа постобработки меняется точность границ объекта.

Основные ошибки модели возникают на сложных фонах или при похожих цветах объекта и фона.

---

## 8. Итоговый вывод

Лучшим вариантом классификации является ResNet18 с partial fine-tuning.

Transfer learning значительно повышает точность и ускоряет обучение.

Сегментация требует корректного выбора метрики и постобработки маски.

Pretrained модели позволяют эффективно решать задачи компьютерного зрения.

---

## 9. Приложение

Дополнительные графики:

[`./artifacts/figures/classification_curves_best.png`](./artifacts/figures/classification_curves_best.png)

[`./artifacts/figures/segmentation_metrics.png`](./artifacts/figures/segmentation_metrics.png)
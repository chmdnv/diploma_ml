# Модель кредитного риск-менеджмента
## Учебный проект для Skillbox ML-junior

Основан на данных https://www.kaggle.com/datasets/mikhailkostin/dlfintechbki<br>
<br>
В ходе моделирования было опробовано 3 модели обучения: LightGBM, HistGradientBoosting и нейронная сеть, реализованная на pytorch<br>
<br>
Итоговая точность лучшей модели на тестовой выборке по метрике roc-auc:
0.7587

## Содержание проекта
**/data** - данные для проекта (загрузить из источника)
* test_predict.csv - предикты для данных из test_data

**/jupyter** - jupyter-notebook файлы:
* collect - загрузка и агрегация данных
* modeling_res - выбор модели (LightGBM, HistGradient)
* modeling_mlp - нейросеть на pytorch
* pipeline - формирование пайплайна с итоговой моделью
* pipeline_test - тестовый предикт на пайплайне
* pipe_utils.py - утилиты для работы пайплайна
* features.pkl - список итоговых фич модели

**/model** - файлы с моделями
<br><br>
## Запуск
Итоговый пайплайн **model/pipe_05.pkl**<br>
Для запуска необходимо импортировать все функции из jupyter/pipe_utils.py<br>
Для десериализации модели *pipe_05* использовать dill (см. пример в *pipeline_test.ipynb*)

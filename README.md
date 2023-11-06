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

**/jupyter** - jupyter-notebook файлы:
* collect - загрузка и агрегация данных
* modeling - выбор модели
* pipeline - пайплайн с итоговой моделью
* pipe_utils.py - утилиты для работы пайплайна

**/model** - файлы с моделями
<br><br>
Итоговый пайплайн **model/hgb_04.pkl**<br>
Для запуска необходимо импортировать все функции из jupyter/pipe_utils.py

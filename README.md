# DA-in-GameDev-lab5
# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Интеграция экономической системы в проект Unity и обучение ML-Agent.

Отчет по лабораторной работе #5 выполнил(а):
- Довгий Вадим Игоревич
- РИ-210942
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 80 |
| Задание 2 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Научиться интегрировать экономическую систему в проект Unity и обучать ML-Agent.

## Задание 1
### Измените параметры файла. yaml-агента и определить какие параметры и как влияют на обучение модели.

1) В самом начале работы я скачал проект и подключил все необходимые пакеты
![2022-12-01_19-02-41](https://user-images.githubusercontent.com/45125347/205074538-56356ece-27f3-41dc-82aa-bbe196b4875c.png)

2) Добавил Economic.yaml в папку с проектом. Ниже предоставлено само содержимое этого файла:
    
```yaml
behaviors:
  Economic:
    trainer_type: ppo   /* Задаёт вид обученияю. Задано обучение с подкреплением */
    hyperparameters:
      batch_size: 1024  /* Обозначает число опытов в за одну итерацию градиентного спуска. */
      buffer_size: 10240    /* Размер необходимого опыта для обновления модели поведения. */
      learning_rate: 3.0e-4 /* Начальная скорость обучения */
      learning_rate_schedule: linear    /* Параметр изменения скорости обучения с течением времени. */
      beta: 1.0e-2  /* Сила регуляриззации энтропии. */
      epsilon: 0.2  /* Влияет на быстроту изменения поведения во время обучения. */
      lambd: 0.95    /* lambd -  это параметр регуляризации. */
      num_epoch: 3       /* Число эпох . */
    network_settings:
      normalize: false  /* Параметр, определяющий автоматическое нормализование обучения. */
      hidden_units: 128  /* Число нейронов в скрытых слоях. */
      num_layers: 2 /* Число скрытых слоёв сети */
    reward_signals:
      extrinsic:
        gamma: 0.99 /* Здесь задаётся коэффициент поощерения, который должен быть меньше единицы. */
        strength: 1.0   /* Коэффициент силы, на который умножается поощерение. */
    checkpoint_interval: 500000
    max_steps: 750000   /* Задаёт максимальное количество повторов симуляции сцены. */
    time_horizon: 64    /* Количество циклов ML агента, хранящихся в буфере до ввода в модель. */
    summary_freq: 5000  /* Здесь задаётся частота сохранения статистики тренировок по шагам. */
    self_play:
      save_steps: 20000
      team_change: 100000
      swap_steps: 10000
      play_against_latest_model_ratio: 0.5
      window: 10
```
3) Далее я запустил виртуальное пространство и запустил обучение программы.

![2022-12-01_19-03-24](https://user-images.githubusercontent.com/45125347/205076351-1121513d-8104-43d1-a41b-a082a15f00e7.png)

4) Установил TensorBoard. Появились следующие графики:
 
![2022-12-01_19-28-21](https://user-images.githubusercontent.com/45125347/205078481-9c50222e-cf41-4071-826c-9048cb3f85c5.png)

![2022-12-01_19-28-30](https://user-images.githubusercontent.com/45125347/205078507-c8c28fa7-fa40-4b2f-9e1e-5740b906326a.png)

5) Далее мы будем изменять различные параметры файла Economic.yaml. Наша задача - добиться максимальной линейности и монотонности графика Cumulative Reward. 

6) Изменил batch_size с 1024 на 2200. Запустил обучение заново и получил новые графики

![2022-12-01_19-47-43](https://user-images.githubusercontent.com/45125347/205082975-a2b3203d-f5e8-4814-9941-b8913a8eee17.png)

График Cumulative изменился.

7) Изменил batch_size с 1024 на 200. Прогнал все пункты заново.

![2022-12-01_19-57-44](https://user-images.githubusercontent.com/45125347/205085547-0bf24f2c-781c-4074-8456-f68805cfb02d.png)

График стал похожим на график в случае 1024.

8) Вернул  batch_size 1024, изменил lambd с 0.95 на 0.85

![2022-12-01_20-07-53](https://user-images.githubusercontent.com/45125347/205089944-f6ffed7c-6cbb-4869-965e-ace734cc319e.png)

график стал более линеен.


10) Изменил num_epoch с 3 на 1

![2022-12-01_20-12-25](https://user-images.githubusercontent.com/45125347/205089803-14b8ef9b-b1fa-46d6-abab-f4e73b2cbf0c.png)

Практически нет изменений.


## Задание 2
### Опишите результаты, выведенные в TensorBoard.

#### Environment
1) Cumulative Reward - среднее общее вознаграждение за эпизод для всех агентов. Увеличивается, когда эпизод обучения успешен. График должен постоянно увеличиваться, но может вести себя скачкообразно.

2) Episode Length - средняя продолжительность эпизода обучения в среде для агентов.

#### Losses
3) Policy Loss - средняя величина функции потери политики, где политика - процесс принятия решений. График должен идти вниз во время успешного эпизода.

4) Value Loss - средняя потеря функции значения. Она моделирует, насколько хорошо агент прогнозирует значение своего следующего состояния. Должна увеличиваться, пока агент обучается, а затем уменьшаться, когда вознаграждение стабилизируется.

#### Policy

5) Entropy - график случайности решений модели. Должен уменьшаться во время успешного эпизода. 
6) Beta - гиперпараметр для настройки Entropy.
7) Epsilon - гиперпараметр, влияет на скорость развития политики.
8) Extrinsic Reward - соответствует среднему совокупному вознаграждению, полученному от окружающей среды за эпизод.
9) Value Estimate - это среднее значение, посещённое всеми состояниями агента. Чтобы отражать увеличение знаний агента, это значение должно расти, а затем стабилизироваться.
10) Learning Rate - показывает величину шага при поиске оптимальной политики. Должен уменьшаться линейно.

#### Self play
11) ELO - показывает силу сети.

## Выводы
В этой лабораторной я научился интегрировать экономическую систему в проект Unity, используя мл агента. Научился выводить графики в TensorBoard и анализировать их.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**

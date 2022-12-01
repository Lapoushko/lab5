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
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      learning_rate_schedule: linear
      beta: 1.0e-2
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3      
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    checkpoint_interval: 500000
    max_steps: 750000
    time_horizon: 64
    summary_freq: 5000
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


Далее буду изменять 5 раз какие-либо параметры файла Economic.yaml. Задача - добиться максимальной монотонности и линейности графика Cumulative Reward.

6) Изменил batch_size с 1024 на 2100. Прогнал все пункты заново.

![image](https://user-images.githubusercontent.com/100460661/204902981-ed6800f5-8e39-4c7d-8904-49d503508ea4.png)

График стал всегда равен 1.

7) Изменил batch_size с 1024 на 300. Прогнал все пункты заново.

![image](https://user-images.githubusercontent.com/100460661/204906752-666444b0-c8c9-4fe1-82b0-0fdc9dbef739.png)

График стал более кривым.

8) Вернул  batch_size 1024, изменил lambd с 0.95 на 0.9

![image](https://user-images.githubusercontent.com/100460661/204908823-3dc96edc-ef01-4e25-9510-2ad3272d8cf2.png)

график стал более линеен.

9) Оставил lambd 0.9 и изменил epsilon с 0.2 на 0.1

![image](https://user-images.githubusercontent.com/100460661/204910295-b7822e91-12c7-4527-b0d4-1f1cbd3fcb0c.png)

Практически нет изменений.

10) Изменил num_epoch с 3 на 1

![image](https://user-images.githubusercontent.com/100460661/204911579-0b373ba2-c0b5-46d4-99e7-7df73578100e.png)

Практически нет изменений.


## Задание 2
### Опишите результаты, выведенные в TensorBoard.

#### Environment
- Cumulative Reward - среднее общее вознаграждение за эпизод для всех агентов. Увеличивается, когда эпизод обучения успешен. График должен постоянно увеличиваться, но может вести себя скачкообразно.

- Episode Length - средняя продолжительность эпизода обучения в среде для агентов.

#### Losses
- Policy Loss - средняя величина функции потери политики, где политика - процесс принятия решений. График должен идти вниз во время успешного эпизода.

- Value Loss - средняя потеря функции значения. Она моделирует, насколько хорошо агент прогнозирует значение своего следующего состояния. Должна увеличиваться, пока агент обучается, а затем уменьшаться, когда вознаграждение стабилизируется.

#### Policy

- Entropy - график случайности решений модели. Должен уменьшаться во время успешного эпизода. 
- Beta - гиперпараметр для настройки Entropy.
- Epsilon - гиперпараметр, влияет на скорость развития политики.
- Extrinsic Reward - соответствует среднему совокупному вознаграждению, полученному от окружающей среды за эпизод.
- Value Estimate - это среднее значение, посещённое всеми состояниями агента. Чтобы отражать увеличение знаний агента, это значение должно расти, а затем стабилизироваться.
- Learning Rate - показывает величину шага при поиске оптимальной политики. Должен уменьшаться линейно.

#### Self play
- ELO - показывает силу сети.

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

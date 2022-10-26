# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил:
- Дубских Семён Николаевич
- РИ210950
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

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
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity.

- Подключил необходимые пакеты для работы с машинным обучением в Unity.
- Создал виртуальное окружение с помощью `conda` и поставил все необходимые зависимости
- Обучил модель на 27 копиях.

![изображение](https://user-images.githubusercontent.com/45539357/197766155-ce4d5303-b32b-4295-bb88-2a0da877d617.png)

В ходе обучения модели заметил, что обучение проходит быстрее, если использовать несколько копий ml-агентов. В итоге модель обучения отлично справилась с выполнением своей задачи, шар не вываливался за границы пола, точно и быстро достигал цели.

![изображение](https://user-images.githubusercontent.com/45539357/197768832-b17c92de-ad3b-4ff0-9234-4b6cae374529.png)

## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети.
```yaml
behaviors:
  RollerBall: # Название модели
    trainer_type: ppo # Алгоритмы обучения модели (ppo, sac, poca)
    hyperparameters: 
      batch_size: 10 # Количество опытов в каждой итерации градиентного спуска
      
      # Количество опытов, которые необходимо собрать перед обновлением политики модели. 
      # Соответствует тому, сколько опыта должно быть собрано, прежде чем модель обучиться или обновиться.
      # Это значение должно быть в несколько раз больше, чем `batch_size`. 
      # Обычно больший размер буфера соответствует более стабильным обновлениям обучения.
      buffer_size: 100
      
      # Начальная скорость обучения для градиентного спуска. 
      # Соответствует силе каждого шага обновления градиентного спуска. 
      # Если обучение нестабильно и вознаграждение постоянно не увеличивается, то это значение следует уменьшать.
      learning_rate: 3.0e-4
      
      # Сила энтропийной регуляризации, которая делает политику «более случайной». 
      # Это гарантирует, что агенты должным образом исследуют пространство действия во время обучения. 
      # Увеличение этого параметра обеспечит выполнение большего количества случайных действий. 
      # Необходимо, чтобы энтропия (измеряемая с помощью TensorBoard) медленно уменьшалась вместе с увеличением вознаграждения. 
      # Если энтропия падает слишком быстро, необходимо увеличить бета. Если энтропия падает слишком медленно, уменьшить.
      beta: 5.0e-4
      
      # Влияет на то, насколько быстро политика может развиваться во время обучения. 
      # Соответствует допустимому порогу расхождения между старой и новой политикой при обновлении градиентного спуска. 
      # Установка небольшого значения этого параметра приведет к более стабильным обновлениям, но также замедлит процесс обучения.
      epsilon: 0.2
      
      # Параметр регуляризации (лямбда) используется при расчете обобщенной оценки преимущества (GAE).
      # Можно рассматривать как то, насколько агент полагается на свою текущую оценку стоимости 
      # при вычислении обновленной оценки стоимости.
      # Низкие значения соответствуют большему полаганию на текущую оценку ценности (что может быть высоким смещением), 
      # а высокие значения соответствуют большему полаганию на фактические вознаграждения, 
      # полученные в среде (что может быть высокой дисперсией). 
      # Правильное значение может привести к более стабильному процессу обучения.
      lambd: 0.99
      
      # Количество проходов через буфер опыта при выполнении оптимизации градиентного спуска. 
      # Уменьшение этого параметра обеспечит более стабильные обновления за счет более медленного обучения.
      num_epoch: 3
      
      # Определяет, как скорость обучения изменяется с течением времени.
      # linear линейно уменьшает learning_rate, достигая 0 при max_steps.
      learning_rate_schedule: linear
      
    network_settings:
    
      # Применяется ли нормализация к входным данным векторного наблюдения. 
      # Эта нормализация основана на скользящем среднем и дисперсии векторного наблюдения. 
      # Нормализация может быть полезна в случаях со сложными задачами непрерывного управления, 
      # но может быть вредна для более простых задач дискретного управления.
      normalize: false
      
      # Количество блоков в скрытых слоях нейронной сети. 
      # Соответствуют количеству единиц в каждом полносвязном слое нейронной сети. 
      # Для простых задач, где правильное действие представляет собой простую комбинацию входных данных наблюдения, 
      # это значение должно быть небольшим. 
      # Для задач, где действие представляет собой очень сложное взаимодействие между переменными наблюдения, 
      # это значение должно быть большим.
      hidden_units: 128
      
      # Количество скрытых слоев в нейронной сети. 
      # Соответствует количеству скрытых слоев после ввода наблюдения или после кодирования CNN визуального наблюдения. 
      # Для простых задач меньше слоев, скорее всего, будут обучать быстрее и эффективнее. 
      # Для более сложных задач управления может потребоваться больше слоев.
      num_layers: 2
      
    reward_signals:
      extrinsic:
        
        # Коэффициент дисконтирования для будущих вознаграждений, поступающих из окружающей среды. 
        # Можно рассматривать как то, как далеко в будущем агент должен заботиться о возможных вознаграждениях. 
        # В ситуациях, когда агент должен действовать в настоящем, 
        # чтобы подготовиться к вознаграждению в отдаленном будущем, это значение должно быть большим. 
        # В случаях, когда вознаграждение является более быстрым, оно может быть меньше. Всегда должно быть строго меньше 1.
        gamma: 0.99
        
        # Фактор, на который умножается вознаграждение, данное средой. 
        # Типичные диапазоны будут варьироваться в зависимости от сигнала вознаграждения.
        strength: 1.0
        
    # Общее количество шагов (т.е. собранных наблюдений и предпринятых действий), 
    # которые необходимо выполнить в среде (или во всех средах при параллельном использовании нескольких)
    # перед завершением процесса обучения. 
    # Если в среде есть несколько агентов с одинаковым именем поведения, все шаги, предпринятые этими агентами, 
    # будут учитывать одно и то же значение max_steps.    
    max_steps: 500000
    
    # Сколько шагов опыта нужно собрать для каждого агента, прежде чем добавить его в буфер опыта.
    # Когда этот предел достигается до конца эпизода, оценка значения используется для прогнозирования 
    # общего ожидаемого вознаграждения из текущего состояния агента. 
    # Таким образом, этот параметр является компромиссом между менее предвзятой, 
    # но более высокой оценкой дисперсии (длинный временной горизонт) и более предвзятой, 
    # но менее разнообразной оценкой (короткий временной горизонт). 
    # В тех случаях, когда в эпизоде есть частые награды или эпизоды непомерно велики, 
    # более идеальным может быть меньшее количество. 
    # Это число должно быть достаточно большим, чтобы охватить все важные действия в последовательности действий агента.
    time_horizon: 64
    
    # Количество опытов, которое необходимо собрать перед созданием и отображением статистики обучения. 
    # Это определяет детализацию графиков в Tensorboard.
    summary_freq: 10000
```

## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

Используя уже обученную модель, сделал так чтобы цель менялась после того как шар достиг первого куба.

```cs
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Random = UnityEngine.Random;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        Target = firstTarget;
    }

    public Transform firstTarget;
    public Transform secondTarget;

    private Transform Target;
    
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        if (Target == firstTarget)
        {
            Target = secondTarget;
        }
        else
        {
            Target = firstTarget;
            secondTarget.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
            firstTarget.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
        }
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    
    public float forceMultiplier = 10;
    
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}

```

![Lab3 - SampleScene - Windows, Mac, Linux - Unity 2021 3 9f1 _DX11_ 2022-10-26 18-18-13](https://user-images.githubusercontent.com/45539357/198037169-b24f2d49-2a93-4eec-8a13-f8ace6fe97e8.gif)

## Выводы

В ходе лабораторной работы я узнал об платформе anaconda и как работать с ней. 
Разобрался в основах MlAgent для Unity и его интеграцией с python и pytorch. 
Понял каким образом происходит обучение модели в ML и как применять эту модель в дальнейшем.

## Приложение
[ссылка на блокнот в google.colab](https://colab.research.google.com/drive/1rXT8cx4VNWsd0JEJmZ3KINgqZyst13XO?usp=sharing)

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**

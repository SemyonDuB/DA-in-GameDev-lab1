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

- Для Python в отчете привести скриншоты с демонстрацией сохранения документа google.collab на свой диск с запуском программы, выводящей сообщение Hello World.

![py2](https://user-images.githubusercontent.com/45539357/190683883-084abc6e-f534-4709-9560-fc7b1f2856b3.png)
![py1](https://user-images.githubusercontent.com/45539357/190683862-7765f787-3ff5-4590-bd2a-77e9eb83089d.png)

- Для Unity в отчете привести скриншоты вывода сообщения Hello World в консоль.

![unity1](https://user-images.githubusercontent.com/45539357/190684142-696f75fd-0cb5-4da3-b4c3-a0916df11d38.png)


## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
- Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

```

- Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py
def model(a, b, x):
  return a*x + b

def loss_function(a, b, x, y):
  num = len(x)
  prediction = model(a, b, x)
  return (0.5/num) * (np.square(prediction - y)).sum()

def optimize(a,b,x,y):
  num = len(x)
  prediction = model(a, b, x)
  da = (1.0/num) * ((prediction - y)*x).sum()
  db = (1.0/num) * ((prediction - y).sum())
  a = a - Lr*da
  b = b - Lr*db
  return a, b

def iterate(a,b,x,y,times):
  for i in range(times):
    a,b = optimize(a,b,x,y)
  return a,b
```

- Шаг 1 Инициализация и модель итеративной оптимизации

```py
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

a,b = iterate(a,b,x,y,1)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```

```
[0.80498831]
[0.82261809]
[0.80792609] [0.82265959] 1557.0714078067256

[<matplotlib.lines.Line2D at 0x7f96a3c28ad0>]
```
![изображение](https://user-images.githubusercontent.com/45539357/190677800-4dc03b90-999c-4804-b027-21c8ab723058.png)

- Шаг 2 На второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации

```py
a,b = iterate(a,b,x,y,2)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a, b, loss)
plt.scatter(x,y)
plt.plot(x, prediction)
```

```
[0.813774] [0.82274216] 1539.995820627625

[<matplotlib.lines.Line2D at 0x7f96a3b8bd50>]
```

![изображение](https://user-images.githubusercontent.com/45539357/190679494-5935150e-6b5c-49a5-8956-91db34d9644b.png)

- Шаг 3 Третья итерация показывает значения параметров, значения потерь и визуализацию после итерации

```py
a,b = iterate(a,b,x,y,3)
prediction = model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
```
[0.83963905] [0.8231069] 1465.759462151177

[<matplotlib.lines.Line2D at 0x7f96a3a28350>]
```

![изображение](https://user-images.githubusercontent.com/45539357/190679961-96573047-0da9-4a9e-900e-fdd195416d7b.png)

- Шаг 4 на четвертой итерации отображаются значения параметров, значения потерь и эффекты визуализации

```py
a,b = iterate(a,b,x,y,4)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```

```
[0.8620235] [0.82342188] 1403.2092232630403

[<matplotlib.lines.Line2D at 0x7f96a3912c90>]
```

![изображение](https://user-images.githubusercontent.com/45539357/190680294-4f12eafa-cf81-48b2-8b12-218693a1f1ca.png)

- Шаг 5 Пятая итерация показывает значение параметра, значение потерь и эффект визуализации после итерации

```py
a,b = iterate(a,b,x,y,5)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x, prediction)
```

```
[0.88922348] [0.82380371] 1329.3209158769337

[<matplotlib.lines.Line2D at 0x7f96a395bf10>]
```
![изображение](https://user-images.githubusercontent.com/45539357/190680630-de0992f3-5f83-4a74-85be-bc48619add55.png)

- Шаг 6 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации

```py
a,b = iterate(a,b,x,y,10000)
prediction = model(a,b,x)
loss = loss_function(a,b,x,y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```

```
[1.74090543] [0.8045214] 191.25586071969795

[<matplotlib.lines.Line2D at 0x7f96a387ad10>]
```

![изображение](https://user-images.githubusercontent.com/45539357/190680931-2862bc40-d571-48a2-9c82-9dd7d584c0ac.png)


## Задание 3

### Изучить код на Python и ответить на вопросы:
- Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

loss может стремиться к нулю при `b = 0` и `x = y`, либо при значениях `x` и `y` стремящихся к нулю:

```py
loss = loss_function(1,0,x, x)

loss = loss_function(2,0,x*10**(-30), y*10**(-30))
```

```
0.0

2.8440000000000007e-58
```



Но при случайных данных и соблюдении условия, что `x != y` и `b != 0` величина `loss` не стремится к нулю:
```py
loss = loss_function(a, b, x, y)
```

```
1166.2355680022215
```

- Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

Lr уменьшает на бесконечно малое число значения `a` и `b`, то есть чем меньше Lr тем меньше изменятся `a` и `b`, чем больше Lr, тем сильнее изменятся `a` и `b`.

При бесконечно малом Lr:

```py
Lr = 10**(-10)
print(f"a = {a},  b = {b}")
print(optimize(a,b,x,y))
```

```
a = [0.9616939],  b = [0.17136955]
a = [0.96169415],  b = [0.17136955]
```

При относительно небольшом Lr:

```py
Lr = 1
print(f"a = {a},  b = {b}")
print(optimize(a,b,x,y))
```

```
a = [0.9616939],  b = [0.17136955]
a = [2477.84371326], b = [34.83103144]
```

## Выводы

В ходе лабораторной работы я научился взаимодействовать с google.colab, просматривать графики и анализировать алгоритм работы программ. Кроме того научился работать в github и ознакомился с основными операторами зыка Python на примере реализации линейной регрессии.

## Приложение
[ссылка на блокнот в google.colab](https://colab.research.google.com/drive/1rXT8cx4VNWsd0JEJmZ3KINgqZyst13XO?usp=sharing)

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**

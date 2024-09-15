"""""""""""""""""""""""""""""""""""""""
network.py
Модуль создания и обучения нейронной сети для распознавания рукописных цифр
с использованием метода градиентного спуска.
Группа:КЭ-120
ФИО:Ращупкин Евгений Владимирович
"""""""""""""""""""""""""""""""""""""""
# Библиотеки
# Стандартные библиотеки
import random  # библиотека функций для генерации случайных значений
# Сторонние библиотеки
import numpy as np  # библиотека функций для работы с матрицами
""" ---Раздел описаний--- """


# определение сигмоидальной функции активации
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


""" --Описание класса Network--"""


class Network(object):  # используется для описания нейронной сети
    def __init__(self, sizes):  # конструктор класса
        # self – указатель на объект класса
        # sizes – список размеров слоев нейронной сети
        self.num_layers = len(sizes)  # задаем количество слоев нейронной сети
        self.sizes = sizes  # задаем список размеров слоев нейронной сети
        # задаем случайные начальные смещения
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # задаем случайные начальные веса связей
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # подсчет выходных сигналов нейронной сети при заданных входных сигналах
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # Стохастический градиентный спуск
    def SGD(
        self,  # указатель на объект класса
        training_data,  # обучающая выборка
        epochs,  # количество эпох обучения
        mini_batch_size,  # размер подвыборки
        eta,  # скорость обучения
        test_data,  # тестирующая выборка
    ):
        # создаем список объектов тестирующей выборки
        test_data = list(test_data)
        # вычисляем длину тестирующей выборки
        n_test = len(test_data)
        # создаем список объектов обучающей выборки
        training_data = list(training_data)
        n = len(training_data)  # вычисляем размер обучающей выборки
        for j in range(epochs):  # цикл по эпохам
            # перемешиваем элементы обучающей выборки
            random.shuffle(training_data)
            # создаем подвыборки
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # цикл по подвыборкам
            for mini_batch in mini_batches:
                # один шаг градиентного спуска
                self.update_mini_batch(mini_batch, eta)
            # смотрим прогресс в обучении
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))

    # Шаг градиентного спуска
    def update_mini_batch(
        self,  # указатель на объект класса
        mini_batch,  # подвыборка
        eta,  # скорость обучения
    ):
        # список градиентов dC/db для каждого слоя (первоначально заполняются нулями)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # список градиентов dC/dw для каждого слоя (первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # послойно вычисляем градиенты dC/db и dC/dw для текущего прецедента (x, y)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # суммируем градиенты dC/db для различных прецедентов текущей подвыборки
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # суммируем градиенты dC/dw для различных прецедентов текущей подвыборки
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # обновляем все веса w нейронной сети
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        # обновляем все смещения b нейронной сети
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    # Алгоритм обратного распространения
    def backprop(
        self,  # указатель на объект класса
        x,  # вектор входных сигналов
        y,  # ожидаемый вектор выходных сигналов
    ):
        # список градиентов dC/db для каждого слоя (первоначально заполняются нулями)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # список градиентов dC/dw для каждого слоя (первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # определение переменных
        # выходные сигналы слоя (первоначально соответствует выходным сигналам 1-го слоя или входным сигналам сети)
        activation = x
        # список выходных сигналов по всем слоям (первоначально содержит только выходные сигналы 1-го слоя)
        activations = [x]
        # список активационных потенциалов по всем слоям (первоначально пуст)
        zs = []
        # прямое распространение
        for b, w in zip(self.biases, self.weights):
            # считаем активационные потенциалы текущего слоя
            z = np.dot(w, activation)+b
            # добавляем элемент (активационные потенциалы слоя) в конец списка
            zs.append(z)
            # считаем выходные сигналы текущего слоя, применяя сигмоидальную функцию активации к активационным потенциалам слоя
            activation = sigmoid(z)
            # добавляем элемент (выходные сигналы слоя) в конец списка
            activations.append(activation)

        # обратное распространение
        # считаем меру влияния нейронов выходного слоя L на величину ошибки (BP1)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta  # градиент dC/db для слоя L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # градиент dC/dw для слоя L (BP4) 
        for l in range(2, self.num_layers):
            z = zs[-l] # активационные потенциалы l-го слоя (двигаемся по списку справа налево)
            sp = sigmoid_prime(z) # считаем сигмоидальную функцию от активационных потенциалов l-го слоя
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # считаем меру влияния нейронов l-го слоя на величину ошибки (BP2)
            nabla_b[-l] = delta # градиент dC/db для l-го слоя (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  # градиент dC/dw для l-го слоя (BP4)
        return (nabla_b, nabla_w)

""" --Конец описания класса Network--"""
""" --- Конец раздела описаний--- """
""" ---Тело программы--- """
net = Network([2, 3, 1])  # создаем нейронную сеть из трех слоев
""" ---Конец тела программы--- """
""" Вывод результата на экран: """
print('Сеть net:')
print('Количетво слоев:', net.num_layers)
for i in range(net.num_layers):
    print('Количество нейронов в слое', i, ':', net.sizes[i])
for i in range(net.num_layers-1):
    print('W_', i+1, ':')
    print(np.round(net.weights[i], 2))
    print('b_', i+1, ':')
    print(np.round(net.biases[i], 2))

"""""""""""""""""""""""""""""""""""""""
network2.py
Модуль создания и обучения нейронной сети для распознавания рукописных цифр на основе метода стохастического градиентного спуска для прямой нейронной сети и стоимостной функции на основе перекрестной энтропии, регуляризации и улучшеннного способа инициализации весов нейронной сети.
Группа:КЭ-120
ФИО:Ращупкин Евгений Владимирович
"""""""""""""""""""""""""""""""""""""""
# Библиотеки
# Стандартные библиотеки
import json  # библиотека для кодирования/декодирования данных/объектов Python
import random  # библиотека функций для генерации случайных значений
import sys  # библиотека для работы с переменными и функциями, имеющими отношение к интерпретатору и его окружению
# Сторонние библиотеки
import numpy as np  # библиотека функций для работы с матрицами

""" ---Раздел описаний--- """


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# определение сигмоидальной функции активации
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Производная сигмоидальной функции
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


""" -- Определение стоимостных функции --"""


# Определение среднеквадратичной стоимостной функции
class QuadraticCost(object):

    @staticmethod
    # Cтоимостная функция
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    # Мера влияния нейронов выходного слоя на величину ошибки
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


# Определение стоимостной функции на основе перекрестной энтропии
class CrossEntropyCost(object):
    @staticmethod
    # Cтоимостная функция
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    # Мера влияния нейронов выходного слоя на величину ошибки
    def delta(z, a, y):
        return (a-y)


""" --Описание класса Network--"""


class Network(object):  # используется для описания нейронной сети
    # конструктор класса
    def __init__(
        self,  # self – указатель на объект класса
        sizes,  # sizes – список размеров слоев нейронной сети
        cost=CrossEntropyCost  # стоимостная функция (по умолчанию будет использоваться функция перекрестной энтропии)
    ):
        # задаем количество слоев нейронной сети
        self.num_layers = len(sizes)
        # задаем список размеров слоев нейронной сети
        self.sizes = sizes
        # метод инициализации начальных весов связей и смещений по умолчанию
        self.default_weight_initializer()
        # задаем стоимостную функцию
        self.cost = cost

    # метод инициализации начальных весов связей и смещений
    def default_weight_initializer(self):
        # задаем случайные начальные смещения
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # задаем случайные начальные веса связей
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        # задаем случайные начальные смещения
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # задаем случайные начальные веса
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # Подсчет выходных сигналов нейронной сети при заданных входных сигналах
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
        lmbda=0.0,  # параметр сглаживания L2-регуляризации
        evaluation_data=None,  # оценочная выборка
        monitor_evaluation_cost=False,  # флаг вывода на экран информа-ции о значении стоимостной функции в процессе обучения, рассчитанном на оценочной выборке
        monitor_evaluation_accuracy=False,  # флаг вывода на экран ин-формации о достигнутом прогрессе в обучении, рассчитанном на оценочной выборке
        monitor_training_cost=False,  # флаг вывода на экран информации о значении стоимостной функции в процессе обучения, рассчитанном на обучающей выборке
        monitor_training_accuracy=False,  # флаг вывода на экран инфор-мации о достигнутом прогрессе в обучении, рассчитанном на обучающей выборке
        ):
        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)
        training_data = list(training_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("--Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("--Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("--Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("--Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    # Шаг градиентного спуска
    def update_mini_batch(
        self,  # указатель на объект класса
        mini_batch,  # подвыборка
        eta,  # скорость обучения
        lmbda,  # параметр сглаживания L2-регуляризации
        n,
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
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
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

        # Определение переменных
        # Выходные сигналы слоя (первоначально соответствует выходным сигналам 1-го слоя или входным сигналам сети)
        activation = x
        # Список выходных сигналов по всем слоям (первоначально содержит только выходные сигналы 1-го слоя)
        activations = [x]
        # Список активационных потенциалов по всем слоям (первоначально пуст)
        zs = []

        # Прямое распространение
        for b, w in zip(self.biases, self.weights):
            # Считаем активационные потенциалы текущего слоя
            z = np.dot(w, activation)+b
            # Добавляем элемент (активационные потенциалы слоя) в конец списка
            zs.append(z)
            # Считаем выходные сигналы текущего слоя, применяя сигмоидальную функцию активации к активационным потенциалам слоя
            activation = sigmoid(z)
            # Добавляем элемент (выходные сигналы слоя) в конец списка
            activations.append(activation)

        # Обратное распространение
        # Считаем меру влияния нейронов выходного слоя L на величину ошибки (BP1)
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        # Градиент dC/db для слоя L (BP3)
        nabla_b[-1] = delta
        # Градиент dC/dw для слоя L (BP4)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            # Активационные потенциалы l-го слоя (двигаемся по списку справа налево)
            z=zs[-l]
            # Считаем сигмоидальную функцию от активационных потенциалов l-го слоя
            sp=sigmoid_prime(z)
            # Считаем меру влияния нейронов l-го слоя  на величину ошибки (BP2)
            delta=np.dot(self.weights[-l+1].transpose(), delta) * sp
            # Градиент dC/db для l-го слоя (BP3)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        # Градиент dC/dw для l-го слоя (BP4)
        return (nabla_b, nabla_w)

    # Оценка прогресса в обучении
    def accuracy(
        self,  # Указатель на объект класса
        data,  # Набор данных (выборка)
        convert=False  # Признак необходимости изменять формат представления результата работы нейронной сети
        ):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    # Оценка прогресса в обучении
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # Вычисление частных производных стоимостной функции по выходным сигналам последнего слоя
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    # Значение функции потерь по всей выборке
    def total_cost(
        self,  # Указатель на объект класса
        data,  # Набор данных (выборка)
        lmbda,  # Параметр сглаживания L2-регуляризации
        convert=False  # Признак необходимости изменять формат представления результата работы нейронной сети
    ):
        cost = 0.0
        data = list(data)
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    # Запись нейронной сети в файл
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    # Загрузка нейронной сети из файла
    def load(filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        cost = getattr(sys.modules[__name__], data["cost"])
        net = Network(data["sizes"], cost=cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net

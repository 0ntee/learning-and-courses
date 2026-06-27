import numpy as np

x = np.array([[1.0, 2.0, -1.0], 
              [0.5, -1.0, 2.0]])

y_true = np.array([[1.0, 0.0], 
                   [0.0, 1.0]])
np.random.seed(42)

class Dense:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.w = np.random.randn(in_features, out_features)
        self.b = np.zeros((1, out_features)) #Обязательно двойные скобки!(Пе__редаём кортеж ?формы?)

        self.dw = None
        self.db = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        z = np.dot(self.x, self.w)
        if self.bias:
            z += self.b
        return z
    
    def backward(self, d_output):
        self.dw = np.dot(self.x.T, d_output)
        if self.bias:
            self.db = np.sum(d_output, axis=0, keepdims=True)
        d_input = np.dot(d_output, self.w.T)
        return d_input

class ReLU:
    def __init__(self):
        self.cache = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        mask = x > 0.0 #Вернёт булевы T/F
        out = x * mask #F обратит в "0", T оставит x = x
        self.cache = mask #Сохранили знак
        return out

    def backward(self, d_output):
        mask = self.cache
        d_input = d_output * mask #Если знак "-" - обнуляем градиент
        self.cache = None
        return d_input

class MSEloss:
    def __init__(self):
        self.cache = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.cache = y_pred, y_true
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        self.y_pred, self.y_true = self.cache
        N = self.y_true.size
        d_output = 2/N * (self.y_pred - self.y_true)
        self.cache = None

        return d_output

#Вводим пайплайн реализации модели
layer = Dense(in_features=3, out_features=2, bias=True)
activation = ReLU()
loss_fn = MSEloss()

epochs = 13
learning_rate = 0.1
batch_size = x.shape[0]

for epoch in range(epochs):
    a1 = layer(x)
    y_pred = activation(a1)
    loss = loss_fn.forward(y_pred, y_true)

    #ОБРАТНЫЙ ХОД
    d_loss = loss_fn.backward()
    d_act = activation.backward(d_loss)
    layer_dwdb = layer.backward(d_act)
    
    layer.w = layer.w - learning_rate * layer.dw
    layer.b = layer.b - learning_rate * layer.db

    if epoch % 2 == 0 or epoch == 13:
        print("Текущая эпоха:")
        print(epoch)
        print("Текущий loss:")
        print(loss)


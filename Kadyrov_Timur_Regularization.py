import numpy as np

np.random.seed(42)

print("Библиотеки загружены.")

N = 60
X = np.linspace(-3, 3, N)

y_true = np.sin(X) * 2 + 0.5
noise = np.random.normal(0, 0.6, N)
y_hat = y_true + noise

X[0], y_true[0], y_hat[0]

class RegularizationMy:
    def __init__(self, alpha=0.1, lr=0.1):
        self.alpha = alpha
        self.lr = lr
        self.weights = [0.5]

    def mse(self, y_true, y_hat):
        s = 0
        for i in range(len(y_true)):
            s += 1/2 * (y_true[i] - y_hat[i]) ** 2
        return s

    def mse_reg(self, y_true, y_hat):
        s = 0
        for i in range(len(y_true)):
            s += 1/2 * (y_true[i] - y_hat[i]) ** 2

        reg = 0
        for i in range(len(self.weights)):
            reg += self.weights[i] ** 2

        return s + self.alpha * reg

    def update_weights(self, y_true, y_hat):
        new_weights = []

        for i in range(len(self.weights)):
            grad = (y_hat[i] - y_true[i]) + 2 * self.alpha * self.weights[i]
            new_w = self.weights[i] - self.lr * grad
            new_weights.append(new_w)

        self.weights = new_weights

    def train(self, y_true, y_hat, epochs=1000):
        self.weights = [0.5] * len(y_true)

        eps = 0.0001
        prev_loss = self.mse_reg(y_true, y_hat)

        print("с Рег (старые веса):", prev_loss)

        for epoch in range(epochs):

            self.update_weights(y_true, y_hat)
            current_loss = self.mse_reg(y_true, y_hat)

            print("epoch", epoch, "| loss:", current_loss)

            if abs(current_loss - prev_loss) < eps:
                print("loss почти не меняется")
                break

            if current_loss < 0.01:
                print("маленькая ошибка")
                break

            prev_loss = current_loss


model = RegularizationMy(alpha=0.1, lr=0.1)

print("без Рег:", model.mse(y_true, y_hat))

model.train(y_true, y_hat)

print("новые веса:", model.weights)
print("новый loss:", model.mse_reg(y_true, y_hat))
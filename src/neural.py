import numpy as np

class Perceptron:
  def __init__(self, hidden_size=1, learning_rate=0.1):
    self.w1 = np.random.randn(2, hidden_size)
    self.b1 = np.random.randn(1, hidden_size)
    self.w2 = np.random.randn(hidden_size, 1)
    self.b2 = np.random.randn(1, 1)
    self.learning_rate = learning_rate

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def feed_forward(self, x):
    self.z1 = np.dot(x, self.w1) + self.b1
    self.a1 = self.sigmoid(self.z1)
    self.z2 = np.dot(self.a1, self.w2) + self.b2
    return self.sigmoid(self.z2)

  def calculate_mse_loss(self, pred, target):
    return 2 * (pred - target) / len(target)

  def calculate_gradient(self, pred, target):
    mse_loss = self.calculate_mse_loss(pred, target)
    sensitivity = self.sigmoid(pred) * (1 - self.sigmoid(pred))
    return (mse_loss, sensitivity)

  def update_weights(self, gradient, inputs):
    mse_loss, sensitivity = gradient

    output_activation = self.a1
    dJ_dW2 = np.dot(output_activation.T, mse_loss * sensitivity)
    dJ_db2 = np.sum(mse_loss * sensitivity, axis=0, keepdims=True)

    hidden_activation_derivative = self.sigmoid(self.z1) * (1 - self.sigmoid(self.z1))
    input_data_reshaped = inputs.reshape(-1, 1)
    dJ_dW1 = np.dot(input_data_reshaped, np.dot(mse_loss * sensitivity, self.w2.T) * hidden_activation_derivative)
    dJ_db1 = np.sum(np.dot(mse_loss * sensitivity, self.w2.T) * hidden_activation_derivative, axis=0, keepdims=True)

    self.w2 -= self.learning_rate * dJ_dW2
    self.b2 -= self.learning_rate * dJ_db2
    self.w1 -= self.learning_rate * dJ_dW1
    self.b1 -= self.learning_rate * dJ_db1

  def fit(self, data, epochs=1000, print_loss_every=100):
    for epoch in range(1, epochs+1):
      for inputs, target in data:
        y = self.feed_forward(inputs)
        gradient = self.calculate_gradient(y, target)
        self.update_weights(gradient, inputs)

      if epoch % print_loss_every == 0:
        print(f'Epoch {epoch}, Loss: {self.calculate_mse_loss(y, target)[0,0]}, Predicted: {self.predict(inputs)}')

  def predict(self, inputs):
    output = self.feed_forward(inputs)
    return 1 if output > 0.5 else 0

def main():
  model = Perceptron()
  training_data = [
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([0])),
    (np.array([1, 0]), np.array([0])),
    (np.array([1, 1]), np.array([1]))
  ]
  model.fit(data=training_data)

  for inputs, target in training_data:
    print(f'Input: {inputs}, Target: {bool(target)}, Predicted: {bool(model.predict(inputs))}')


if __name__ == "__main__":
  main()
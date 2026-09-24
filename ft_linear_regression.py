import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from scipy import stats



class LinearRegression():
  def __init__(self, d_x, d_y, lr = 0.0001, epochs = 100):
    self.d_x = d_x
    self.d_y = d_y
    self.lr = lr
    self.epochs = epochs

    rng = np.random.default_rng(42)
    self.a = rng.uniform(40000, 300000)
    self.b = rng.uniform(2000, 9000)


    print(self.a, self.b)




  def predict(self):
    # return pd.Series((self.a * x + self.b), dtype=np.int64, name="Predict")
    y_pred = self.a * self.d_x + self.b
    y_pred.name = 'Predict'
    # print(self.pred)

    return y_pred

  def MeanSquaredError(self, y_pred):
    lossError = (y_pred - self.d_y) ** 2
    lossError.name = 'MSE(lossError)'

    return lossError

  def grad(self, y_pred):
    da = 2 * (self.d_x * (y_pred - self.d_y))
    db = 2 * (y_pred - self.d_y)

    self.a -= self.lr * da
    self.b -= self.lr * db

    return (da, db)


  def vis(self, y_pred):
    # print(self.pred)
    plt.plot(self.d_x, self.d_y, 'ro')
    plt.plot(self.d_x, y_pred, 'b-')
    # plt.xlim(0, 300000)
    # plt.ylim(0, 90000)
    plt.show()

    
    

  def optimizer(self):
    y_pred = 0
    for _ in range(self.epochs):
      y_pred = self.predict()
      lossError = self.MeanSquaredError(y_pred)
      self.grad(y_pred)


      # print(lossError)

    self.vis(y_pred)








      









def main():
  df = pd.read_csv('data.csv')

  model = LinearRegression(df.km, df.price)
  model.optimizer()
  # print(model.a, model.b)
  



  # slope, intercept, r, p, std_err = stats.linregress(df.km, df.price)

  # def myfunc(x):
  #   return slope * x + intercept

  # mymodel = list(map(myfunc, df.km))

  # plt.plot(df.km, df.price, 'ro')
  # plt.plot(df.km, mymodel)
  # plt.show()




if __name__ == '__main__':
  main()

import csv
import math
import json




def mean(lst: list):
    return sum(lst) / len(lst)

def std(lst: list):
    mean_ = mean(lst)
    s = 0
    for l in lst:
        s += (l + mean_) ** 2
    variance = s / len(lst)
    return math.sqrt(variance)
     


class LinearRegression:

    def __init__(self, x, y, learning_rate=0.01, epochs=1000):

        # self.x = x
        self.y = y

        # Normalize mileage to make gradient descent stable (z-score)
        self.x_mean = mean(x)
        self.x_std = std(x)

        self.x =  [((val - self.x_mean) / self.x_std) for val in x]

        self.learning_rate = learning_rate
        self.epochs = epochs

        # theta0 and theta1 start at 0
        self.theta0 = 0.0
        self.theta1 = 0.0


    def estimate_price(self):
        """
        estimatePrice(mileage) = theta0 + theta1 * mileage
        """
        return [ (self.theta0 + self.theta1 * val) for val in self.x ]

    def train(self):

        print("Training...")

        m = len(self.x)

        for _ in range(self.epochs):

            # Prediction using current theta0/theta1
            predictions = self.estimate_price()



            # Error for every training example
            errors = [ predictions[i] - self.y[i] for i in range(len(self.y))]
            


            # Formula from the subject
            tmp_theta0 = (
                self.learning_rate
                * (1 / m)
                * sum(errors)
            )

            tmp_theta1 = (
                self.learning_rate
                * (1 / m)
                * sum([errors[i] * self.x[i] for i in range(len(self.x))])
            )

            # IMPORTANT:
            # Both theta values are updated simultaneously
            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1

    def save(self, filename="theta.json"):

        data = {
            "theta0": float(self.theta0),
            "theta1": float(self.theta1),
            "x_mean": float(self.x_mean),
            "x_std": float(self.x_std)
        }

        with open(filename, "w") as file:
            json.dump(data, file, indent=4)

        print(f"Model saved to {filename}")




def read_csv(filename):
    x = []
    y = []

    with open(filename, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x.append(float(row["km"]))
            y.append(float(row["price"]))

    return x, y


def main():

    try:
        x, y = read_csv('data.csv')
        

        model = LinearRegression(
            x,
            y,
            learning_rate=0.1,
            epochs=1000
        )


        model.train()

        print("Training completed.")
        print(f"theta0 = {model.theta0}")
        print(f"theta1 = {model.theta1}")

        model.save()

    except FileNotFoundError:
        print("Error: data.csv not found.")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()

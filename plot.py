import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_model(filename="theta.json"):

    with open(filename, "r") as file:
        data = json.load(file)

    return (
        data["theta0"],
        data["theta1"],
        data["x_mean"],
        data["x_std"]
    )


def main():

    try:
        data = pd.read_csv("data.csv")

        theta0, theta1, x_mean, x_std = load_model()

        x = data["km"].astype(float)
        y = data["price"].astype(float)

        # Normalize X
        x_normalized = (x - x_mean) / x_std

        # Predictions
        predictions = theta0 + theta1 * x_normalized

        # Sort for a clean line
        # order = np.argsort(x)

        # plt.figure(figsize=(10, 6))

        # Real data
        plt.scatter(
            x,
            y,
            label="Training data"
        )

        # Regression line
        plt.plot(
            x,
            predictions,
            label="Regression line"
        )

        plt.xlabel("Mileage (km)")
        plt.ylabel("Price")
        plt.title("Linear Regression")

        plt.legend()
        plt.grid(True)

        plt.show()

    except FileNotFoundError:
        print("Error: data.csv or theta.json not found.")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
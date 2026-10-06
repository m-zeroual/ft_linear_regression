import json
import numpy as np
import pandas as pd


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

        # Normalize mileage
        x_normalized = (x - x_mean) / x_std

        # Prediction
        predictions = theta0 + theta1 * x_normalized


        # Additional information
        mae = np.mean(
            np.abs(y - predictions)
        )

        mse = np.mean(
            (y - predictions) ** 2
        )

        rmse = np.sqrt(mse)

        print()
        print(f"MAE:  {mae:.2f}")
        print(f"MSE:  {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")

    except FileNotFoundError:
        print("Error: data.csv or theta.json not found.")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
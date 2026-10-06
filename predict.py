import json


def load_model(filename="theta.json"):

    try:
        with open(filename, "r") as file:
            data = json.load(file)

        return (
            data["theta0"],
            data["theta1"],
            data["x_mean"],
            data["x_std"]
        )

    except FileNotFoundError:
        print("Warning: theta.json not found.")
        print("theta0 and theta1 will be initialized to 0.")

        return 0.0, 0.0, 0.0, 1.0

    except (KeyError, json.JSONDecodeError):
        print("Error: invalid theta.json")
        return 0.0, 0.0, 0.0, 1.0





def main():

    theta0, theta1, x_mean, x_std = load_model()

    try:

        mileage = float(input("Enter mileage: "))

        if mileage < 0:
            print("Mileage cannot be negative.")
            return

        # Normalize mileage exactly as during training
        normalized_mileage = (
            mileage - x_mean
        ) / x_std

        # Hypothesis:
        price = theta0 + theta1 * normalized_mileage

        # A car price cannot be negative
        price = max(0, price)

        print(f"Estimated price: {price:.2f}")

    except ValueError:
        print("Please enter a valid number.")


if __name__ == "__main__":
    main()
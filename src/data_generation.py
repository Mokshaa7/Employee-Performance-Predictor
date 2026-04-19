import pandas as pd
import numpy as np

def generate_data(n=500, path="data/employee_data.csv"):
    np.random.seed(42)

    data = pd.DataFrame({
        "Age": np.random.randint(22, 60, n),
        "Experience": np.random.randint(1, 20, n),
        "Department": np.random.choice(["HR", "IT", "Sales"], n),
        "Salary": np.random.randint(30000, 120000, n),
        "Training_Hours": np.random.randint(10, 100, n),
        "Projects": np.random.randint(1, 10, n),
        "Attendance": np.random.randint(70, 100, n)
    })

    data["Performance"] = np.where(
        (data["Experience"] > 10) &
        (data["Training_Hours"] > 50) &
        (data["Attendance"] > 85),
        "High", "Low"
    )

    data.to_csv(path, index=False)
    print("Dataset created at:", path)

if __name__ == "__main__":
    generate_data()
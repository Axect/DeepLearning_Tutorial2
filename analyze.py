import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import scienceplots
import beaupy
from rich.console import Console

from util import (
    select_project,
    select_group,
    select_seed,
    select_device,
    load_model,
    load_data,
    load_study,
    load_best_model,
)


def main():
    # Test run
    console.print("[bold green]Analyzing the model...[/bold green]")
    console.print("Select a project to analyze:")
    project = select_project()
    console.print("Select a group to analyze:")
    group_name = select_group(project)
    console.print("Select a seed to analyze:")
    seed = select_seed(project, group_name)
    console.print("Select a device:")
    device = select_device()
    model, config = load_model(project, group_name, seed)
    model = model.to(device)

    df_true = pl.read_parquet("./data/true.parquet")
    x_true = df_true["x"].to_numpy()
    y_true = df_true["y"].to_numpy()

    df_data = pl.read_parquet("./data/data.parquet")
    x_data = df_data["x"].to_numpy()
    y_data = df_data["y"].to_numpy()

    model.eval()
    x_test = torch.linspace(0, 1, 3000).unsqueeze(1)
    with torch.no_grad():
        y_test = model(x_test.to(device)).cpu().numpy().squeeze()
    x_test = x_test.cpu().numpy().squeeze()

    rmse = np.sqrt(np.mean((y_true - y_test) ** 2))

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, '.', label="Data", color='blue', markersize=2, markeredgewidth=0, alpha=0.5)
        ax.plot(x_true, y_true, label="True", color='red')
        ax.plot(x_test, y_test, '--', label="Predicted", color='orange')
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"RMSE: {rmse:.4e}")
        ax.autoscale(tight=True)
        fig.savefig("plot.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    console = Console()
    main()

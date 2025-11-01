import matplotlib.pyplot as plt
import csv

def plot_learning_curve(csv_path: str, output_path: str = "learning_curve.png"):
    iterations = []
    losses = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row["iteration"]))
            losses.append(float(row["train_loss"]))

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, losses, label="Training Loss", color="blue", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Transformer LM – Learning Curve on TinyStories", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved learning curve to {output_path}")

if __name__ == "__main__":
    plot_learning_curve("loss_log.csv")

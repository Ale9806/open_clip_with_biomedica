import pandas as pd

def add_experiment_to_csv(
    csv_path: str,
    user: str = "",
    data_range: str = "",
    train_num_samples: int = 1000000,
    accum_freq: int = 2,
    lr_scheduler: str = "cosine",
    dataset_type: str = "webdataset",
    learning_rate: str = "2e-5",
    beta1: float = 0.9,
    beta2: float = 0.95,
    warmup_steps: int = 1000,
    weight_decay: float = 0.2,
    batch_size: int = 64,
    epochs: int = 100,
    num_workers: int = 1,
    model: str = "RN50",
    precision: str = "fp16",
    log_steps: int = 1,
    seed: int = 0,
    logs_dir: str = "./logs_clip_rn50/",
    pretrained: str = "openai",
    report_to: str = "wandb",
    wandb_project_name: str = "open-biomed-clip"
) -> None:
    """
    Adds a new experiment configuration to a CSV file. If the CSV doesn't exist, 
    it creates a new one. The configuration includes hyperparameters, model setup, 
    and logging preferences.

    Args:
        csv_path (str): Path to the CSV file where the experiment configuration will be added.
        user (str): Name of the user adding the experiment.
        data_range (str): Range of data used for the experiment.
        train_num_samples (int): Number of training samples used.
        accum_freq (int): Gradient accumulation frequency.
        lr_scheduler (str): Learning rate scheduler type.
        dataset_type (str): Type of the dataset (e.g., 'webdataset').
        learning_rate (str): Learning rate for the optimizer.
        beta1 (float): Beta1 parameter for the optimizer.
        beta2 (float): Beta2 parameter for the optimizer.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        weight_decay (float): Weight decay (L2 regularization) value.
        batch_size (int): Batch size used for training.
        epochs (int): Number of epochs to train the model.
        num_workers (int): Number of worker threads for data loading.
        model (str): Model architecture name (e.g., 'RN50').
        precision (str): Precision mode (e.g., 'fp32').
        log_steps (int): Number of steps between logging outputs.
        seed (int): Random seed for reproducibility.
        logs_dir (str): Directory for saving logs.
        pretrained (str): Pretrained model source (e.g., 'openai').
        report_to (str): Reporting tool (e.g., 'wandb').
        wandb_project_name (str): Name of the project on Weights & Biases.

    Returns:
        None
    """
    # Load the existing CSV or create a new DataFrame
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    # Create a new row as a dictionary
    new_row = {
        "user": user,
        "data_range": data_range,
        "train_num_samples": train_num_samples,
        "accum_freq": accum_freq,
        "lr_scheduler": lr_scheduler,
        "dataset_type": dataset_type,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_workers": num_workers,
        "model": model,
        "precision": precision,
        "log_steps": log_steps,
        "seed": seed,
        "logs_dir": logs_dir,
        "pretrained": pretrained,
        "report_to": report_to,
        "wandb_project_name": wandb_project_name,
    }

    # Append the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    df.to_csv(csv_path, index=False)

    print(f"Experiment added to {csv_path}.")


if __name__=='__main__':

    csv_path = "overfit.csv"

    add_experiment_to_csv(csv_path, user="ale9806", batch_size=64, data_range="{000000..000001}")
    add_experiment_to_csv(csv_path, user="ale9806", batch_size=32, data_range="{000000..000001}")



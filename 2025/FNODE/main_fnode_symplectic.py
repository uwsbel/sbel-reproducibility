# FNODE/main_fnode_symplectic.py
# Train a grad-output model with symplectic integrators (Störmer–Verlet, Yoshida4, Fukushima6).

import os
import sys
import time
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.optim as optim


def setup_logging(log_file_path: str | None = None) -> None:
    """Setup logging with optional file output, avoiding duplicate handlers."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        handlers.insert(0, logging.FileHandler(log_file_path, mode="w"))

    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)


setup_logging()
logger = logging.getLogger("FNODE_Symplectic_Main")


# Ensure Model directory is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
model_package_dir = os.path.join(script_dir, "Model")
if model_package_dir not in sys.path:
    sys.path.insert(0, model_package_dir)


try:
    from Model.Data_generator import generate_dataset, generate_slider_crank_dataset
    from Model.utils import set_seed, get_output_paths, save_model_state, load_model_state, calculate_model_parameters
    from Model.model import (
        MBDNODE_Symplectic,
        train_fnode_symplectic_grad,
        _masses_for_test_case,
    )
    from Model.integrator import (
        sep_stormer_verlet_multiple_body,
        yoshida4_multiple_body,
        fukushima6_multiple_body,
    )
except ImportError as e:
    logger.error(f"Failed to import Model modules: {e}", exc_info=True)
    raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FNODE-style Hamiltonian gradients with symplectic rollout loss"
    )

    parser.add_argument(
        "--test_case",
        type=str,
        default="Single_Mass_Spring",
        choices=[
            "Single_Mass_Spring",
            "Single_Mass_Spring_Damper",
            "Double_Pendulum",
            "Triple_Mass_Spring_Damper",
            "Slider_Crank",
            "Cart_Pole",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--generate_new_data", action="store_true", default=True)
    parser.add_argument("--data_dt", type=float, default=0.01)
    parser.add_argument("--data_total_steps", type=int, default=3000)
    parser.add_argument("--train_ratio", type=float, default=0.1)

    parser.add_argument(
        "--integrator",
        type=str,
        default="stormer_verlet",
        choices=["stormer_verlet", "yoshida4", "fukushima6"],
        help="Symplectic integrator used in rollout loss and testing.",
    )
    parser.add_argument(
        "--symp_step_delay",
        type=int,
        default=2,
        help="Number of symplectic steps to unroll for each training sample.",
    )

    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh"])
    parser.add_argument("--initializer", type=str, default="xavier", choices=["xavier", "kaiming"])

    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="exponential",
        choices=["none", "exponential", "step", "cosine"],
    )
    parser.add_argument("--lr_decay_rate", type=float, default=0.98)
    parser.add_argument("--lr_decay_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--out_time_log", type=int, default=1)
    parser.add_argument("--skip_train", action="store_true", default=False)
    parser.add_argument("--model_load_filename", type=str, default="FNODE_best.pkl")

    return parser.parse_args()


def _load_dataset(test_case: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset_path = os.path.join(".", "dataset", test_case)
    s_train_file = os.path.join(dataset_path, "s_train.csv")
    t_train_file = os.path.join(dataset_path, "t_train.csv")
    s_test_file = os.path.join(dataset_path, "s_test.csv")
    t_test_file = os.path.join(dataset_path, "t_test.csv")

    s_train_df = pd.read_csv(s_train_file)
    s_test_df = pd.read_csv(s_test_file)
    t_train_df = pd.read_csv(t_train_file, header=None, names=["time"])
    t_test_df = pd.read_csv(t_test_file, header=None, names=["time"])

    # For Slider Crank, we only use theta and omega columns if present.
    if test_case == "Slider_Crank":
        if "theta_0_2pi" in s_train_df.columns and "omega" in s_train_df.columns:
            s_train_np = s_train_df[["theta_0_2pi", "omega"]].values
            s_test_np = s_test_df[["theta_0_2pi", "omega"]].values
        else:
            s_train_np = s_train_df.values[:, 1:3] if s_train_df.columns[0] in ["idx", "Unnamed: 0"] else s_train_df.values[:, 0:2]
            s_test_np = s_test_df.values[:, 1:3] if s_test_df.columns[0] in ["idx", "Unnamed: 0"] else s_test_df.values[:, 0:2]
    else:
        s_train_np = s_train_df.values[:, 1:] if s_train_df.columns[0] in ["idx", "Unnamed: 0"] else s_train_df.values
        s_test_np = s_test_df.values[:, 1:] if s_test_df.columns[0] in ["idx", "Unnamed: 0"] else s_test_df.values

    t_train_np = t_train_df["time"].values
    t_test_np = t_test_df["time"].values

    # Keep data on CPU; training helper moves to model device.
    s_train = torch.tensor(s_train_np, dtype=torch.float32, device="cpu")
    t_train = torch.tensor(t_train_np, dtype=torch.float32, device="cpu")
    s_test = torch.tensor(s_test_np, dtype=torch.float32, device="cpu")
    t_test = torch.tensor(t_test_np, dtype=torch.float32, device="cpu")
    return s_train, t_train, s_test, t_test


def main() -> int:
    args = parse_arguments()
    set_seed(args.seed)

    current_device = torch.device(args.device)

    # Logging to file
    base_log_dir = os.path.join(os.getcwd(), "log")
    log_dir = os.path.join(base_log_dir, args.test_case)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(os.path.join(log_dir, f"fnode_symplectic_{args.integrator}.log"))
    global logger
    logger = logging.getLogger("FNODE_Symplectic_Main")

    model_type_for_paths = f"FNODE_symplectic_{args.integrator}"
    output_paths = get_output_paths(args.test_case, model_type_for_paths)
    os.makedirs(output_paths["model"], exist_ok=True)
    os.makedirs(output_paths["results"], exist_ok=True)
    os.makedirs(output_paths["figures"], exist_ok=True)

    # Data generation
    dataset_path = os.path.join(".", "dataset", args.test_case)
    s_train_file = os.path.join(dataset_path, "s_train.csv")
    t_train_file = os.path.join(dataset_path, "t_train.csv")
    data_gen_needed = args.generate_new_data or not os.path.exists(s_train_file) or not os.path.exists(t_train_file)

    num_steps_for_train_segment = int(args.data_total_steps * args.train_ratio)
    if data_gen_needed:
        logger.info(f"Generating dataset for {args.test_case} (dt={args.data_dt}, steps={args.data_total_steps})")
        if args.test_case == "Slider_Crank":
            generate_slider_crank_dataset(
                total_num_steps=args.data_total_steps * 10,
                train_num_steps=num_steps_for_train_segment * 10,
                dt=args.data_dt,
                root_dir=".",
                seed=args.seed,
            )
        else:
            generate_dataset(
                test_case=args.test_case,
                numerical_methods="rk4",
                dt=args.data_dt,
                num_steps=args.data_total_steps,
                seed=args.seed,
                gen_train_num_steps=num_steps_for_train_segment,
                train_split_ratio=args.train_ratio,
                output_root_dir=".",
                save_to_file=True,
            )

    # Load dataset
    s_train, t_train, s_test, t_test = _load_dataset(args.test_case)
    s_full = torch.cat([s_train, s_test], dim=0)
    t_full = torch.cat([t_train, t_test], dim=0)

    raw_state_dim = int(s_train.shape[1])
    if raw_state_dim % 2 != 0:
        raise ValueError(f"State dimension must be even. Got {raw_state_dim}")
    num_bodys = raw_state_dim // 2

    logger.info(f"Loaded data: train={tuple(s_train.shape)} test={tuple(s_test.shape)} bodies={num_bodys}")

    # Build grad-output model: outputs [dH/dq, dH/dp]
    model = MBDNODE_Symplectic(
        num_bodys=num_bodys,
        layers=args.layers,
        width=args.hidden_size,
        activation="tanh" if args.activation == "tanh" else "relu",
        initializer="xavier" if args.initializer == "xavier" else "xavier",
    ).to(current_device)
    calculate_model_parameters(model)

    # Optimizer + scheduler
    optimizer_class = optim.AdamW if args.optimizer == "adamw" else optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.lr)

    scheduler = None
    if args.lr_scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_decay_steps, eta_min=args.lr * 1e-4)

    best_model_path = os.path.join(output_paths["model"], args.model_load_filename)

    if not args.skip_train:
        logger.info("=== Training (supervised dH targets; no integrator) ===")
        train_params = {
            "epochs": args.epochs,
            "outime_log": args.out_time_log,
            "grad_clip": args.grad_clip,
            "batch_size": args.batch_size,
            "grad_fd_order": 4,
            "grad_smooth_boundaries": False,
        }
        start = time.time()
        model, loss_history = train_fnode_symplectic_grad(
            model=model,
            s_train=s_train,
            t_train=t_train,
            train_params=train_params,
            optimizer=optimizer,
            scheduler=scheduler,
            output_paths=output_paths,
            test_case=args.test_case,
            symplectic_integrator=args.integrator,
        )
        logger.info(f"Training done in {time.time() - start:.2f}s")
        save_model_state(model, output_paths["model"], model_filename="FNODE_final.pkl")

    if os.path.exists(best_model_path):
        load_model_state(model, output_paths["model"], model_filename=args.model_load_filename, current_device=current_device)
        logger.info(f"Loaded best model: {best_model_path}")

    # === Testing rollout using the chosen symplectic integrator ===
    logger.info("=== Testing ===")
    model.eval()

    methods = {
        "stormer_verlet": sep_stormer_verlet_multiple_body,
        "yoshida4": yoshida4_multiple_body,
        "fukushima6": fukushima6_multiple_body,
    }
    integration_method = methods[args.integrator]

    # Convert initial state and ground truth to canonical form
    masses = _masses_for_test_case(args.test_case, num_bodys, current_device)

    # dt for rollout: prefer uniform spacing from t_full
    if len(t_full) < 2:
        raise ValueError("Need at least 2 time points")
    dt = float((t_full[1] - t_full[0]).item())
    num_steps = int(len(t_full))

    s0 = s_full[0].to(current_device).view(num_bodys, 2)  # [q, v]
    s0_qp = s0.clone()
    s0_qp[:, 1] = s0[:, 1] * masses

    def dH_dq(x_qp, model=None):
        body_n = x_qp.shape[0]
        pred = model(x_qp.view(-1))
        return pred[:body_n]

    def dH_dp(x_qp, model=None):
        body_n = x_qp.shape[0]
        pred = model(x_qp.view(-1))
        return pred[body_n:2 * body_n]

    with torch.no_grad():
        if args.integrator in {"stormer_verlet", "yoshida4"}:
            traj_qp = integration_method(
                bodys=s0_qp,
                dH_dq=dH_dq,
                dH_dp=dH_dp,
                num_steps=num_steps,
                time_step=dt,
                if_final_state=False,
                current_device=current_device,
                model=model,
            )
        else:
            traj_qp = integration_method(
                bodys=s0_qp,
                dH_dq=dH_dq,
                dH_dp=dH_dp,
                num_steps=num_steps,
                time_step=dt,
                if_final_state=False,
                device=current_device,
                model=model,
            )

    # traj_qp: [T, num_bodys, 2] with [q, p] -> convert back to [q, v]
    traj_qv = traj_qp.clone()
    traj_qv[:, :, 1] = traj_qp[:, :, 1] / masses.view(1, -1)
    pred = traj_qv.view(num_steps, -1).detach().cpu()

    gt = s_full[:num_steps].cpu()
    mse = torch.mean((pred - gt) ** 2).item()
    logger.info(f"Rollout MSE (full): {mse:.6e}")

    # Save predictions and mse
    pred_path = os.path.join(output_paths["results"], f"pred_{args.integrator}.csv")
    pd.DataFrame(pred.numpy()).to_csv(pred_path, index=False)
    pd.DataFrame({"mse": [mse]}).to_csv(os.path.join(output_paths["results"], "test_metrics.csv"), index=False)
    logger.info(f"Saved predictions to {pred_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

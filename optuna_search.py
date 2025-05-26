import os
import json
import wandb
import optuna
import argparse
from functools import partial
from optuna.storages import RDBStorage
from optuna.samplers import NSGAIISampler

from config import model_config, search_space
from data import get_data_config, get_datasets
from utils.set_paths import set_base_paths
from train_val import train_val
from test import test, get_beta_threshold, test_ad
from hw_converter.convert2hw import convert2hw

from utils.set_paths import set_base_paths
from utils.plots import plot_pareto, plot_pareto_rmse, plot_pareto_acc
from optuna_utils import vivado_runner, radiant_runner


def objective(trial, args):
    try:
        enable_qat = args.enable_qat

        # search space
        quant_bits = (
            trial.suggest_int("quant_bits", **search_space["quant_bits"])
            if enable_qat
            else None
        )
        batch_size = trial.suggest_int("batch_size", **search_space["batch_size"])
        lr = trial.suggest_float("lr", **search_space["lr"])
        d_model = trial.suggest_int("d_model", **search_space["d_model"])

        # Get datasets
        data_config = get_data_config(
            args.task_flag,
            args.data_flag,
        )
        (
            train_dataset,
            val_dataset,
            test_dataset,
            target_scaler,
            num_in_features,
            num_out_features,
        ) = get_datasets(data_config)

        # set model configs
        model_params = model_config.copy()
        model_params.update(
            {
                "d_model": d_model,
                "num_in_features": num_in_features,
                "num_out_features": num_out_features,
                "window_size": (
                    int(data_config["window_size"] / data_config["downsample_rate"])
                    if args.task_flag == "classification"
                    else data_config["window_size"]
                ),
                "enable_qat": args.enable_qat,
            }
        )
        if args.enable_qat:
            model_params.update(
                {
                    "name": "network",
                    "quant_bits": quant_bits,
                    "do_int_forward": False,
                }
            )

        # set exp_save_path
        given_timestamp = ""
        exp_save_dir, fig_save_dir, log_save_dir = set_base_paths(
            "train", f"{args.exp_base_save_dir}", given_timestamp
        )
        trial.set_user_attr("timestamp", str(exp_save_dir).split("/")[-1])

        # set up wandb
        wandb.init(
            project=args.wandb_project_name,
            mode=args.wandb_mode,
            config=args,
        )
        wandb.log(
            {
                "batch_size": batch_size,
                "lr": lr,
            }
        )
        # execute training and validation
        best_val_loss, pareto_acc, _, _, _ = train_val(
            task_flag=args.task_flag,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_params=model_params,
            batch_size=batch_size,
            target_scaler=target_scaler,
            lr=lr,
            num_epochs=args.num_epochs,
            exp_save_dir=exp_save_dir,
            fig_save_dir=fig_save_dir,
            log_save_dir=log_save_dir,
        )

        # execute test
        test_config = {
            "task_flag": args.task_flag,
            "data_flag": args.data_flag,
            "test_dataset": test_dataset,
            "model_params": model_params,
            "target_scaler": target_scaler,
            "exp_mode": "train",
            "batch_size": batch_size,
            "exp_save_dir": exp_save_dir,
            "fig_save_dir": fig_save_dir,
            "log_save_dir": log_save_dir,
        }
        if args.task_flag == "anomaly_detection":
            beta, threshold = get_beta_threshold(
                data_flag=args.data_flag,
                data_config=get_data_config(args.task_flag, args.data_flag),
                val_dataset=val_dataset,
                model_params=model_params,
                exp_mode="train",
                exp_save_dir=exp_save_dir,
            )
            test_config.update(
                {
                    "beta": beta,
                    "threshold": threshold,
                }
            )
            test_fn = test_ad
        else:
            test_config.update({"target_scaler": target_scaler})
            test_fn = test

        test_metrics = test_fn(**test_config)

        # get test_loss_denorm
        if args.task_flag == "forecasting":
            test_accuracy = test_metrics["test_denorm_rmse"]
        elif (
            args.task_flag == "classification" or args.task_flag == "anomaly_detection"
        ):
            test_accuracy = test_metrics["test_f1"]
        else:
            raise ValueError(f"Unsupported task_flag: {args.task_flag}")
        trial.set_user_attr(
            "test_accuracy",
            round(float(test_accuracy), 3) if test_accuracy is not None else None,
        )

        if enable_qat:
            # integer-only inference
            model_params["do_int_forward"] = True
            int_test_metrics = test_fn(**test_config)
            if args.task_flag == "forecasting":
                int_test_accuracy = int_test_metrics["int_test_denorm_rmse"]
            elif (
                args.task_flag == "classification"
                or args.task_flag == "anomaly_detection"
            ):
                int_test_accuracy = int_test_metrics["int_test_f1"]
            else:
                raise ValueError(f"Unsupported task_flag: {args.task_flag}")
            trial.set_user_attr(
                "int_test_accuracy",
                (
                    round(float(int_test_accuracy), 3)
                    if int_test_accuracy is not None
                    else None
                ),
            )

            # convert to target and harware simulation
            if args.enable_hw_simulation:

                # convert to target hw
                convert2hw(
                    test_dataset=(
                        test_dataset[0]
                        if args.task_flag == "anomaly_detection"
                        else test_dataset
                    ),
                    subset_size=args.subset_size,
                    model_params=model_params,
                    exp_save_dir=exp_save_dir,
                    target_hw=args.target_hw,
                )

                if args.target_hw == "amd":
                    # get vivado report info
                    hw_metrics = vivado_runner(
                        base_dir=os.path.abspath(
                            os.path.join(exp_save_dir, "hw", "amd")
                        ),
                        top_module="network",
                        trial=trial,
                    )
                elif args.target_hw == "lattice":
                    # get lattice report info
                    hw_metrics = radiant_runner(
                        base_dir=os.path.abspath(
                            os.path.join(exp_save_dir, "hw", "lattice")
                        ),
                        top_module="network",
                        trial=trial,
                    )
                else:
                    raise ValueError(f"Unsupported target_hw: {args.target_hw}")

                power = hw_metrics["power"]
                latency = hw_metrics["latency"]
                energy = hw_metrics["energy"]
                trial.set_user_attr(
                    "power",
                    round(float(power), 3) if power is not None else None,
                )
                trial.set_user_attr(
                    "latency",
                    round(float(latency), 3) if latency is not None else None,
                )
                trial.set_user_attr(
                    "energy",
                    round(float(energy), 3) if energy is not None else None,
                )

        trial.set_user_attr(
            "best_val_loss",
            round(float(best_val_loss), 3) if best_val_loss is not None else None,
        )
        trial.set_user_attr(
            "pareto_acc",  # the actual metric depends on the task
            (round(float(pareto_acc), 3) if pareto_acc is not None else None),
        )

        return best_val_loss, hw_metrics[args.optuna_hw_target]

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data configs
    parser.add_argument(
        "--task_flag",
        type=str,
        choices=["forecasting", "anomaly_detection", "classification"],
    )
    parser.add_argument("--data_flag", type=str)
    parser.add_argument("--downsampling_factor", type=int, default=None)
    parser.add_argument("--data_approach", type=str, default=None)
    parser.add_argument("--target_person", type=str, default=None)

    # wandb configs
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument(
        "--wandb_mode", type=str, choices=["online", "offline", "disabled"]
    )
    # model configs
    parser.add_argument(
        "--enable_qat", action="store_true", help="normal training or qat"
    )
    # exp configs
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--exp_base_save_dir", type=str)
    # quantization configs
    parser.add_argument("--quant_bits", type=int, choices=[4, 6, 8])
    # hw conversion configs
    parser.add_argument(
        "--subset_size",
        type=int,
        help="Number of samples to use for HW simulation",
    )
    parser.add_argument(
        "--enable_hw_simulation",
        action="store_true",
        help="Do HW simulation for quantized model",
    )
    # optuna configs
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument(
        "--optuna_hw_target",
        type=str,
        choices=["power", "latency", "energy"],
    )
    parser.add_argument(
        "--target_hw",
        type=str,
        choices=["amd", "lattice"],
        help="Target hardware for conversion",
    )
    args = parser.parse_args()

    os.makedirs(args.exp_base_save_dir, exist_ok=True)
    db_name = f"{args.task_flag}_{args.data_flag}_{args.enable_qat}.db"
    db_path = os.path.join(args.exp_base_save_dir, db_name)
    storage = RDBStorage(f"sqlite:///{db_path}")

    study = optuna.create_study(
        directions=["minimize", "minimize"],  # [val_loss, hw_metric]
        sampler=NSGAIISampler(),
        storage=storage,
        load_if_exists=True,
        study_name=f"{args.task_flag}_{args.data_flag}_{args.enable_qat}",
    )
    study.optimize(
        partial(objective, args=args), n_trials=args.n_trials, catch=(Exception,)
    )
    # Save the all trial to JSON
    with open(
        f"{args.exp_base_save_dir}/all_trials_{args.task_flag}_{args.data_flag}_{args.enable_qat}.json",
        "w",
    ) as f:
        all_trials_data = [
            {
                "trial": t.number,
                "val_loss": (
                    round(t.values[0], 3)
                    if t.values is not None and t.values[0] is not None
                    else None
                ),
                "hw_metric": (
                    round(t.values[1], 3)
                    if t.values is not None and t.values[1] is not None
                    else None
                ),
                "params": t.params,
                "user_attrs": t.user_attrs,
                "state": t.state.name,  # Save the state (e.g., COMPLETE, FAIL)
            }
            for t in study.trials
        ]
        json.dump(all_trials_data, f, indent=4)

    # Save Pareto front trials to JSON
    with open(
        f"{args.exp_base_save_dir}/pareto_trials_{args.task_flag}_{args.data_flag}_{args.enable_qat}.json",
        "w",
    ) as f:
        pareto_data = [
            {
                "trial": t.number,
                "val_loss": (
                    round(t.values[0], 3)
                    if t.values is not None and t.values[0] is not None
                    else None
                ),
                "hw_metric": (
                    round(t.values[1], 3)
                    if t.values is not None and t.values[1] is not None
                    else None
                ),
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.best_trials
        ]
        json.dump(pareto_data, f, indent=4)

    # Plot Pareto front
    all_trials = study.trials
    pareto_trials = study.best_trials

    pareto_losses = []
    pareto_hw_metrics = []
    for t in pareto_trials:
        if t.values is not None and t.values[0] is not None and t.values[1] is not None:
            pareto_losses.append(t.values[0])
            pareto_hw_metrics.append(t.values[1])

    non_pareto_losses = []
    non_pareto_hw_metrics = []
    for t in all_trials:
        if (
            t not in pareto_trials
            and t.values is not None
            and t.values[0] is not None
            and t.values[1] is not None
        ):
            non_pareto_losses.append(t.values[0])
            non_pareto_hw_metrics.append(t.values[1])

    plot_pareto(
        pareto_losses,
        pareto_hw_metrics,
        non_pareto_losses,
        non_pareto_hw_metrics,
        save_path=f"{args.exp_base_save_dir}/pareto_plot_{args.task_flag}_{args.data_flag}_{args.enable_qat}.pdf",
    )

    # if args.task_forecasting  == "forecasting", then use pareto denorm_rmse to draw a plot
    if args.task_flag == "forecasting":
        pareto_denorm_rmse = []
        for t in pareto_trials:
            if (
                t.user_attrs is not None
                and "pareto_acc" in t.user_attrs
                and t.user_attrs["pareto_acc"] is not None
            ):
                pareto_denorm_rmse.append(t.user_attrs["pareto_acc"])

        non_pareto_denorm_rmse = []
        for t in all_trials:
            if (
                t not in pareto_trials
                and t.user_attrs is not None
                and "pareto_acc" in t.user_attrs
                and t.user_attrs["pareto_acc"] is not None
            ):
                non_pareto_denorm_rmse.append(t.user_attrs["pareto_acc"])

        plot_pareto_rmse(
            pareto_denorm_rmse,
            pareto_hw_metrics,
            non_pareto_denorm_rmse,
            non_pareto_hw_metrics,
            save_path=f"{args.exp_base_save_dir}/pareto_plot_{args.task_flag}_{args.data_flag}_{args.enable_qat}_denorm_rmse.pdf",
        )
    elif args.task_flag == "classification":

        pareto_accs = []
        for t in pareto_trials:
            if (
                t.user_attrs is not None
                and "pareto_acc" in t.user_attrs
                and t.user_attrs["pareto_acc"] is not None
            ):
                pareto_accs.append(t.user_attrs["pareto_acc"])

        non_pareto_accs = []
        for t in all_trials:
            if (
                t not in pareto_trials
                and t.user_attrs is not None
                and "pareto_acc" in t.user_attrs
                and t.user_attrs["pareto_acc"] is not None
            ):
                non_pareto_accs.append(t.user_attrs["pareto_acc"])

        plot_pareto_acc(
            pareto_accs,
            pareto_hw_metrics,
            non_pareto_accs,
            non_pareto_hw_metrics,
            save_path=f"{args.exp_base_save_dir}/pareto_plot_{args.task_flag}_{args.data_flag}_{args.enable_qat}_f1.pdf",
        )

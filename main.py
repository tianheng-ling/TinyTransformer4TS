import wandb
import argparse

from config import model_config
from data import get_data_config, get_datasets
from utils.set_paths import set_base_paths
from train_val import train_val
from test import test, get_beta_threshold, test_ad
from hw_converter.convert2hw import convert2hw


def main(args):
    # get datasets
    data_config = get_data_config(
        args.task_flag,
        args.data_flag,
    )
    data_config.update({"window_size": data_config["window_size"]})
    (
        train_dataset,
        val_dataset,
        test_dataset,
        target_scaler,
        num_in_features,
        num_out_features,
    ) = get_datasets(data_config)

    # set exp_save_path
    exp_save_dir, fig_save_dir, log_save_dir = set_base_paths(
        args.exp_mode, args.exp_base_save_dir, args.given_timestamp
    )

    # set model
    model_params = model_config.copy()
    model_params.update(
        {
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
                "quant_bits": args.quant_bits,
                "do_int_forward": False,
            }
        )

    # train or test or int_inference
    if args.exp_mode == "train":

        # set up wandb
        wandb.init(project=args.wandb_project_name, mode=args.wandb_mode, config=args)

        train_val(
            task_flag=args.task_flag,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_params=model_params,
            batch_size=args.batch_size,
            target_scaler=target_scaler,
            lr=args.lr,
            num_epochs=args.num_epochs,
            exp_save_dir=exp_save_dir,
            fig_save_dir=fig_save_dir,
            log_save_dir=log_save_dir,
        )
        wandb.log(data_config)

    test_config = {
        "task_flag": args.task_flag,
        "data_flag": args.data_flag,
        "test_dataset": test_dataset,
        "model_params": model_params,
        "target_scaler": target_scaler,
        "exp_mode": args.exp_mode,
        "batch_size": args.batch_size,
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
            exp_mode=args.exp_mode,
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

    test_fn(**test_config)

    if args.enable_qat:
        model_params["do_int_forward"] = True
        test_fn(**test_config)
        for target_hw in ["amd", "lattice"]:
            convert2hw(
                test_dataset=(
                    test_dataset[0]
                    if args.task_flag == "anomaly_detection"
                    else test_dataset
                ),
                subset_size=args.subset_size,
                model_params=model_params,
                exp_save_dir=exp_save_dir,
                target_hw=target_hw,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # wandb config
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_mode", type=str)

    # data config
    parser.add_argument(
        "--task_flag",
        type=str,
        choices=["forecasting", "anomaly_detection", "classification"],
    )
    parser.add_argument("--data_flag", type=str)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--downsampling_factor", type=int, default=None)
    parser.add_argument("--data_approach", type=str, default=None)
    parser.add_argument("--target_person", type=str, default=None)

    # experiment config
    parser.add_argument("--exp_mode", type=str, choices=["train", "test"])
    parser.add_argument("--exp_base_save_dir", type=str)
    parser.add_argument("--given_timestamp", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)

    # model_config

    # quantization config
    parser.add_argument(
        "--quant_bits",
        type=int,
        choices=[4, 6, 8],
        help="Number of bits to quantize the model",
    )
    parser.add_argument("--enable_qat", action="store_true", help="Quantize the model")

    # hw simulation config
    parser.add_argument(
        "--subset_size",
        type=int,
        help="Number of samples to use for HW simulation",
    )

    main(args=parser.parse_args())

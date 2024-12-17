import argparse
from ml_collections import config_flags
from sac import train_sac
from xirl.utils import set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run RL with environment rewards")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (inherits from base_configs/rl.py)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save training logs and checkpoints",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    args = parser.parse_args()

    config = config_flags.load_config(args.config)

    set_seed(args.seed)

    setup_logging(log_dir=args.log_dir)

    train_sac(
        config=config,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()


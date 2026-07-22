"""Launch the QLoRA training job on SageMaker.

The machine and GPU choice both live in code, here:
  --instance ml.g5.8xlarge   1x A10G 24GB  (default; right shape for 14B QLoRA)
  --instance ml.g5.12xlarge  4x A10G       (pass a different --gpu per job, or
                                            launch 4 jobs for parallel experiments)

Run from the repo root (Windows is fine; the job itself runs on Linux):
  python scripts/06_launch_sagemaker.py --role-arn arn:aws:iam::<acct>:role/<sagemaker-role>

Spot instances + max_run cap keep a stuck job from burning the budget; the job
terminates and stops billing the moment train.py exits (Playbook Rule 7 analog:
nobody has to remember to stop this pod).
"""
import argparse
import os

import sagemaker
from sagemaker.pytorch import PyTorch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default="ml.g5.8xlarge")
    p.add_argument("--gpu", default="0", help="CUDA index inside the box (matters on 12xlarge)")
    p.add_argument("--train_file", default="rehearsal_train.jsonl")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--job_name", default=None, help="set per-experiment, e.g. exp-a-20-paraphrases")
    p.add_argument("--role-arn", dest="role_arn", default=None,
                   help="SageMaker execution role ARN (required when launching from a laptop)")
    p.add_argument("--no-spot", action="store_true", help="use on-demand instead of spot")
    p.add_argument("--multi-gpu", dest="multi_gpu", action="store_true",
                   help="train on every GPU of the instance via torchrun (DDP)")
    args = p.parse_args()

    sess = sagemaker.Session()
    role = args.role_arn or sagemaker.get_execution_role()
    bucket = sess.default_bucket()
    job_name = args.job_name or "finance-qlora"

    data_path = os.path.join("data", "train", args.train_file)
    s3_train = sess.upload_data(data_path, key_prefix="finance-llm/data")

    est = PyTorch(
        entry_point="train.py",
        source_dir=os.path.join("scripts", "train_job"),
        role=role,
        instance_type=args.instance,
        instance_count=1,
        framework_version="2.5.1",
        py_version="py311",
        hyperparameters={
            "train_file": args.train_file,
            "gpu": args.gpu,
            "epochs": args.epochs,
        },
        environment={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
        use_spot_instances=not args.no_spot,
        max_run=4 * 3600,
        max_wait=8 * 3600 if not args.no_spot else None,
        base_job_name=job_name,
        distribution=({"torch_distributed": {"enabled": True}}
                      if args.multi_gpu else None),
        # Live-sync every checkpoint to S3 as it is written; survives spot
        # reclaims and lets train.py resume from the last checkpoint.
        checkpoint_s3_uri=f"s3://{bucket}/finance-llm/checkpoints/{job_name}",
        volume_size=100,  # GB; keeping every checkpoint needs the headroom
    )
    est.fit({"train": os.path.dirname(s3_train) or s3_train})
    print("Model artifacts:", est.model_data)


if __name__ == "__main__":
    main()

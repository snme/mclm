import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup

from utils import AtomisticLanguageDataset, custom_collate_fn, FullAtomisticLanguageDataset
from alm import AtomisticLanguageModel
import wandb


def train(args):
    device = torch.device("cuda")
    use_wandb = not args.disable_wandb

    model = AtomisticLanguageModel(
        llm_name='Qwen/Qwen3-8B',
        atomistic_model_name='orb_v3_direct_20_omat',
        device=device,
    )
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    if args.train_csv_path is not None:
        dataset = AtomisticLanguageDataset(
            tokenizer=model.tokenizer,
            db_path=args.db_path,
            csv_path=args.train_csv_path,
            thinking=args.thinking,
            max_num_tokens=args.max_num_tokens,
        )
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    else:
        train_dataset = FullAtomisticLanguageDataset(
            tokenizer=model.tokenizer,
            split='train',
            parent_folder=args.data_parent_path,
            thinking=args.thinking,
            max_num_tokens=args.max_num_tokens,
        )
        val_dataset = FullAtomisticLanguageDataset(
            tokenizer=model.tokenizer,
            split='validation',
            parent_folder=args.data_parent_path,
            thinking=args.thinking,
            max_num_tokens=args.max_num_tokens,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    optim = torch.optim.AdamW(
        model.projector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    num_epochs = args.num_epochs
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = min(2000, int(0.03 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_epoch = args.start_epoch
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "projector_state_dict" in checkpoint:
            model.projector.load_state_dict(checkpoint["projector_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optim.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if args.start_epoch == 0:
                start_epoch = int(checkpoint.get("epoch", 0))
            print(
                f"Resumed training state from {args.resume_from_checkpoint} "
                f"at epoch {start_epoch}."
            )
        else:
            model.projector.load_state_dict(checkpoint)
            print(
                "Loaded projector-only checkpoint. "
                "Optimizer/scheduler state not found; restarting those states."
            )

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )

    # training loop
    for epoch in tqdm(
        range(start_epoch, num_epochs),
        desc="Training",
        dynamic_ncols=True,
        position=0,
    ):
        model.train()
        model.llm.eval()
        model.atomistic_model.eval()
        optim.zero_grad(set_to_none=True)

        for step, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Train Epoch {epoch}",
            dynamic_ncols=True,
            leave=False,
            position=1,
        ):
            row_batch = batch['atom_rows']
            input_ids = [ids.to(device) for ids in batch["input_ids"]]
            labels = [lab.to(device) for lab in batch["labels"]]
            attention_mask = [mask.to(device) for mask in batch["attention_mask"]]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(row_batch, input_ids, attention_mask, labels)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            if step % 100 == 0:
                print(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")

            global_step = epoch * len(train_dataloader) + step
            if use_wandb and global_step % args.log_every == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "global_step": global_step,
                })

            if step % args.eval_every == 0:
                model.eval()
                model.llm.eval()
                model.atomistic_model.eval()

                val_loss = 0
                for val_step, batch in tqdm(
                    enumerate(val_dataloader),
                    desc="Validation",
                    total=len(val_dataloader),
                    dynamic_ncols=True,
                    leave=False,
                    position=1,
                ):
                    row_batch = batch['atom_rows']
                    input_ids = [ids.to(device) for ids in batch["input_ids"]]
                    labels = [lab.to(device) for lab in batch["labels"]]
                    attention_mask = [mask.to(device) for mask in batch["attention_mask"]]

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            outputs = model(row_batch, input_ids, attention_mask, labels)
                            val_loss += outputs.loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                print(f"Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}")

                if use_wandb:
                    wandb.log({
                        "val/loss": avg_val_loss,
                        "epoch": epoch,
                    })

                torch.save(
                    model.projector.state_dict(),
                    args.model_save_path.replace(".pt", f"_step={step}.pt"),
                )
                if args.checkpoint_save_path is not None:
                    checkpoint_path = args.checkpoint_save_path.format(epoch=epoch + 1)
                    torch.save(
                        {
                            "projector_state_dict": model.projector.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch + 1,
                            "global_step": (epoch + 1) * len(train_dataloader),
                        },
                        checkpoint_path,
                    )

                model.train()
                model.llm.eval()
                model.atomistic_model.eval()

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=None) # default="/home/sathya/Data/LLM4Mat-Bench/data/oqmd/train.db"
    parser.add_argument("--train_csv_path", type=str, default=None) # default="/home/sathya/Data/LLM4Mat-Bench/data/oqmd/train.csv") # if training on a single ALM dataset
    parser.add_argument("--data_parent_path", type=str, default='/home/sathya/Data/LLM4Mat-Bench/data') # if training on a parent directory of ALM datasets
    parser.add_argument("--model_save_path", type=str, default="/home/sathya/Documents/mclm/alm/checkpoint_model.pt")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="alm-pretrain")
    parser.add_argument("--max_num_tokens", type=int, default=2048)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_save_path", type=str, default=None)
    args = parser.parse_args()
    train(args)

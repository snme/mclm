import argparse
import torch
from torch.utils.data import DataLoader
from ase.db import connect
import polars as pl

from utils import AtomisticLanguageDataset, custom_collate_fn
from alm import AtomisticLanguageModel


def generate_from_row(model, row, max_new_tokens=512, temperature=0.6, top_p=0.95):
    """Generate a description given an ASE database row."""
    device = model.device

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # 1. Encode atoms
            atomistic_features, n_atoms = model.encode_atoms([row])
            atomistic_features = torch.split(atomistic_features, n_atoms)

            # 2. Build prompt (same messages as the dataset)
            messages = [
                {"role": "system", "content": "You are an expert at materials science and atomistic structure."},
                {"role": "user", "content": "<atoms>\nDescribe the structure of this material."},
            ]
            prompt_ids = model.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            input_ids_tensor = torch.tensor([prompt_ids], device=device)

            # 3. Stitch via _merge_embeddings (same path as training forward pass)
            embed_layer = model.llm.get_input_embeddings()
            text_embeds = [embed_layer(input_ids_tensor[0])]
            dummy_labels = [torch.full((input_ids_tensor.shape[1],), -100, dtype=torch.long, device=device)]
            attn_mask = [torch.ones(input_ids_tensor.shape[1], dtype=torch.long, device=device)]

            inputs_embeds, _, attention_mask = model._merge_embeddings(
                text_embeds, atomistic_features, [input_ids_tensor[0]], dummy_labels, attn_mask
            )

            # 4. Generate
            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.3,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
            )

    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate(args):
    device = torch.device("cuda")

    # Load model (no DDP)
    model = AtomisticLanguageModel(
        llm_name="Qwen/Qwen3-8B",
        atomistic_model_name="orb_v3_direct_20_omat",
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "projector_state_dict" in checkpoint:
        model.projector.load_state_dict(checkpoint["projector_state_dict"])
    else:
        model.projector.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Load dataset (same object as training)
    dataset = AtomisticLanguageDataset(
        tokenizer=model.tokenizer,
        db_path=args.db_path,
        csv_path=args.csv_path,
        thinking=False,
        max_num_tokens=args.max_num_tokens,
    )
    df = pl.read_csv(args.csv_path)

    # Iterate over samples
    for i in range(min(args.n_samples, len(dataset))):
        sample = dataset[i]
        row = sample["atom_rows"][0]
        ground_truth = df[i]["description"][0]
        formula = row.toatoms().get_chemical_formula()

        response = generate_from_row(
            model, row,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        print(f"\n{'='*60}")
        print(f"Sample {i} | {formula}")
        print(f"{'='*60}")
        print(f"GROUND TRUTH:\n{ground_truth[:300]}")
        print(f"---")
        print(f"GENERATED:\n{response[:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--db_path", type=str, default="/tmp/oqmd.db")
    parser.add_argument("--csv_path", type=str, default="/tmp/train.csv")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_num_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    evaluate(args)
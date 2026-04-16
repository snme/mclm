from vllm import LLM
import pandas as pd
import argparse
import time
import torch
from tqdm import tqdm


def main(args):
    # read in a test csv
    time_start = time.time()
    print(f"Reading data from {args.data}")

    df = pd.read_csv(args.data)
    filtered_df = df[['oqmd_id', 'e_form', 'cif_structure', 'description']]
    filtered_df = filtered_df.set_index('oqmd_id', drop=True)
    # oqmd_id is already the index after set_index; no need to drop it again

    ids = list(filtered_df.index)
    descriptions = list(filtered_df['description'])
    cifs = list(filtered_df['cif_structure'])

    filtered_df = filtered_df.drop(columns=['description', 'cif_structure'])
    del df

    print(f"Loaded {len(ids)} descriptions")
    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.2f} seconds")

    time_start = time.time()
    print(f"Loading LLM")
    llm = LLM(
        model="Qwen/Qwen3-Embedding-8B",
        runner="pooling",
        dtype="float16",
        enforce_eager=True,
        tensor_parallel_size=4,       # 4 GPUs per node
        # distributed_executor_backend="ray",
    )
    print(f"LLM loaded")
    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.2f} seconds")

    time_start = time.time()
    print(f"Caching embeddings")

    batch_size = 25
    all_embeddings = []
    all_ids = []

    for start_ind in tqdm(range(0, len(descriptions), batch_size)):
        batch = descriptions[start_ind : start_ind + batch_size]
        # truncate string lengths for now. will not be a problem for models with
        # larger context windows
        batch = [b[:10000] for b in batch]
        try:
            outs = llm.embed(batch)
        except:
            print("Error in batch", start_ind)
            continue
        
        # fix: get length of the embedding vector, not the output object
        max_len = max(len(inp) for inp in batch)
        print(max_len)

        all_embeddings.extend([output.outputs.embedding for output in outs])
        all_ids.extend([ids[start_ind + i] for i in range(len(batch))])

        if start_ind % (batch_size * 10) == 0:
            embs = torch.tensor(all_embeddings).cpu()
            torch.save(embs, args.emb_output)

            with open(args.ids_output, 'w') as f:
                f.write('\n'.join([str(i) for i in all_ids]))

    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.2f} seconds")

    embs = torch.tensor(all_embeddings).cpu()
    torch.save(embs, args.emb_output)

    with open(args.ids_output, 'w') as f:
        f.write('\n'.join([str(i) for i in all_ids]))

    print(f"Saved {len(all_embeddings)} embeddings to {args.emb_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/sathyae/orcd/pool/train.csv")
    parser.add_argument("--emb_output", type=str, default="/home/sathyae/orcd/pool/descriptions_embeddings.pt")
    parser.add_argument("--ids_output", type=str, default="/home/sathyae/orcd/pool/descriptions_ids.txt")
    args = parser.parse_args()
    main(args)

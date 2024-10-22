import argparse
from main import get_all_classes, get_examples
import pickle
import os


def main():
    parser = argparse.ArgumentParser(description="Train Model for Matres")
    parser.add_argument("--dataset", type=str, help="add a dataset", required=True, choices=["MATRES", "TIMELINE", "MATRES_NEW", "TB-DENSE"])
    parser.add_argument("--shots", type=int, nargs="*", default=[3,6])
    args = parser.parse_args()

    data_base_path = f"data/{args.dataset}"
    splits = {}
    for file in os.listdir(data_base_path):
        root_name = file.split(".")[0]
        if root_name in ["train", "valid", "test"]:
            print(os.path.join(data_base_path, file))
            with open(os.path.join(data_base_path, file), "rb") as reader:
                splits[root_name] = pickle.load(reader)

    shots = args.shots
    n_runs = 10
    n_classes = len(get_all_classes(splits))
    print(get_all_classes(splits))
    save_dir = f"./saved_examples/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    for n_shots in shots:
        print("-"*100)
        print(f"n_shots: {n_shots}")
        samples = [[] for _ in range(n_runs)]
        save_examples_file = os.path.join(save_dir, f"samples_{n_shots}_shot.pkl")
        examples = get_examples(splits["train"], n_shots*n_runs)
        for run in range(n_runs):
            for rel_id in range(n_classes):
                # start = rel_id*n_runs*n_shots + run*n_shots
                run_offset = (n_shots/n_classes)
                rel_type_offset = ((n_runs*n_shots)/n_classes)
                start = rel_type_offset*rel_id + run_offset*run
                assert start - int(start) == 0, f"{start}, {int(start)}"
                start = int(start)
                sl = slice(start, start + int(run_offset))
                samples[run].extend(examples[sl])
        # check examples unique
        id_set = set()
        for sample in samples:
            print([f"{doc.d_id}_{e1.e_id}_{e2.e_id}_{rel_type}" for doc, e1, e2, rel_type in sample])
            for doc, e1, e2, rel_type in sample:
                id_set.add(f"{doc.d_id}_{e1.e_id}_{e2.e_id}_{rel_type}")
        print(sum([len(sample) for sample in samples]))
        assert len(id_set) == sum([len(sample) for sample in samples])
        print("-"*100)

        with open(save_examples_file, "wb") as f:
            pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

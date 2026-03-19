import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    args = p.parse_args()
    print(f"[INFO] Pré-processamento do dataset: {args.dataset} (placeholder)")

if __name__ == '__main__':
    main()

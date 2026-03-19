import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    print(f"[INFO] Avaliação com config {args.config} (placeholder)")

if __name__ == '__main__':
    main()

import torch

def test_gpu(n=100,m=50,k=10):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on: {device}")

        A = torch.rand(n, n, device=device)
        B = torch.rand(n, m, device=device)

        for _ in range(k):
            C = A @ B
            D = torch.inverse(A)

        print("GPU computation succeeded!")

    except Exception as e:
        print(f"GPU test failed: {e}")

if __name__ == "__main__":
    test_gpu()

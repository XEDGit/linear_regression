import matplotlib.pyplot as plt
import csv
from concurrent.futures import ProcessPoolExecutor

# Helpers and functions
TRANSLATE = lambda x: x / 1_000_000 if x > 1 else int(x * 1_000_000)
regress = lambda t0, t1, x: t1 * x + t0
err = lambda x, y, t0, t1: regress(t0, t1, x) - y


def main():

    epochs = 1_000_000

    # Get data
    file = open("data.csv", "r")
    data = [
        [TRANSLATE(int(d[0])), TRANSLATE(int(d[1]))] for d in list(csv.reader(file))[1:]
    ]

    # Generate learning rates (LRs) array in linear space
    n = 100
    start = 0.095
    end = 0.5
    step = (end - start) / (n - 1)
    lrs = [start + v * step for v in range(n)]

    # Plot labels
    plt.xlabel("epochs")
    plt.ylabel("loss")

    # Train with all LRs
    with ProcessPoolExecutor(max_workers=50) as tpe:
        result = list(
            tpe.map(train, [i for i in range(n)], [data] * n, [epochs] * n, lrs)
        )

    # Extract weights and losses
    ws = [r[0] for r in result]
    all_losses = [r[1] for r in result]

    # Find best
    best = [0, 0, 1000]

    # Plot losses
    for i, lr in enumerate(lrs):
        losses = all_losses[i]
        plt.plot(losses, label=f"{lr:.6f}")
        if best[2] > losses[-1]:
            best = [i, lr, losses[-1]]
    plt.show()

    # New plot window
    plt.figure()

    # Plot data
    plt.scatter([d[0] for d in data], [d[1] for d in data])

    # Pick best weights and estimate against ground truth
    print("best: ", best)
    w = ws[best[0]]
    for k, p, pred in [[km, price, regress(*w, km)] for km, price in data]:
        print(
            f"eval: km: {TRANSLATE(k)}, price: {TRANSLATE(p)}, predicted: {TRANSLATE(pred)}"
        )

    # Plot estimations in linear space against data points
    n = 100
    start = 1
    end = -1
    step = (end - start) / (n - 1)
    x = [start + v * step for v in range(n)]
    y = [regress(*w, i) for i in x]
    plt.plot(x, y)
    plt.show()


def train(index, data, epochs, lr):
    losses = [0.0] * epochs
    w = (0.0, 0.0)
    e = 0
    try:
        for e in range(epochs):
            s0 = [err(km, price, *w) for km, price in data]
            s1 = [err(km, price, *w) * km for km, price in data]

            tmp_w = (
                w[0] - (1 / len(data)) * sum(s0) * lr,
                w[1] - (1 / len(data)) * sum(s1) * lr,
            )
            losses[e] = sum([err(km, price, *tmp_w) ** 2 for km, price in data]) / len(
                data
            )
            if e >= 10 and losses[e] > losses[e - 10]:
                break
            w = tmp_w
    except OverflowError:
        pass

    losses = losses[: e + 1]

    print(f"{index} {lr=:.7f} {losses[-1]=}")
    return w, losses


if __name__ == "__main__":
    main()

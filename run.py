import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

TRANSLATE = lambda x: x / 1_000_000 if x > 1 else int(x * 1_000_000)
regress = lambda t0, t1, x: t1 * x + t0
err = lambda x, y, t0, t1: regress(t0, t1, x) - y


def main():

    file = open('data.csv', 'r')
    data = [[TRANSLATE(int(d[0])), TRANSLATE(int(d[1]))] for d in list(csv.reader(file))[1:]]

    epochs = 100000

    # Gen lrs
    n = 100
    start = 1e-7
    end = 1e-3
    step = (end - start) / (n - 1)
    lrs = [start + v * step for v in range(n)]
    # lrs = [1]

    plt.xlabel('epochs')
    plt.ylabel('loss')

    smallest = [0, 0, 1000]

    last_loss = 0


    for i, lr in enumerate(tqdm(lrs, 'lr')):
        losses = train(data, epochs, lr)
        plt.plot(losses, label=f'{lr:.6f}')

        tmp_smallest = smallest
        if losses[-1] < smallest[2]:
            tmp_smallest = [i, lr, losses[-1]]

        tqdm.write(f'min: {smallest}')
        tqdm.write(f'diff w last: {losses[-1] - last_loss}')
        tqdm.write(f'diff w best: {losses[-1] - smallest[2]}')

        smallest = tmp_smallest


    plt.legend()
    plt.show()


def train(data, epochs, lr):
    losses = [.0] * epochs
    w = (.0, .0)
    e = 0
    try:
        for e in range(epochs):
            s0 = [err(km, price, *w) for km, price in data]
            s1 = [err(km, price, *w) * km for km, price in data]

            tmp_w = (w[0] - (1/len(data)) * sum(s0) * lr, w[1] - (1/len(data)) * sum(s1) * lr)
            losses[e] = sum([err(km, price, *tmp_w) ** 2 for km, price in data]) / len(data)
            if e >= 10 and losses[e] > losses[e-10]:
                break
            w = tmp_w
    except OverflowError:
        pass

    losses = losses[:e+1]

    # for k, p, pred in [[km, price, regress(*w, km)] for km, price in data]:
    #     print(f"km: {TRANSLATE(k)}, price: {TRANSLATE(p)}, predicted: {TRANSLATE(pred)}")

    tqdm.write(f'{lr=:.7f} {losses[-1]=}')
    return losses

if __name__ == '__main__':
    main()
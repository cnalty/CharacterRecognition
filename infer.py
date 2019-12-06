import torch.utils.data
from torch import nn
from model import CharNet
import pandas as pd

BATCH_SIZE = 2048


def main():
    # Load model and data
    model = CharNet(10)
    model.load_state_dict(torch.load("wieghts.pth.tar"))
    model.cuda()
    print(model)

    test = pd.read_csv('KannadaNumerals/test.csv')


    with open("predictions.csv", 'w+') as f:
        f.write("ImageId,Label\n")
        infer(model, test, f)


def infer(model, test, out_file):
    model.eval()

    for index, row in test.iterrows():
        data = row.to_numpy(dtype=float) / 255.0
        #data = data[1:]
        data = data.reshape(28,28)
        data = torch.tensor([[data]]).float()
        data = data.cuda()

        output = model(data)
        result = output.data.cpu().numpy().argmax()
        out_file.write(str(index + 1) + "," + str(result) + "\n")








if __name__ == "__main__":
    main()

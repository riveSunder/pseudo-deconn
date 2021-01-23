import numpy as np
import torch

import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from deconn import Deconn

import matplotlib.pyplot as plt

def get_dataloader(root_path, batch_size=512, num_workers=2, image_size=64):
    dataset = dset.ImageFolder(root=root_path,\
            transform=transforms.Compose([\
            transforms.Resize(image_size),\
            transforms.ToTensor(),\
        ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,\
                    shuffle=True, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":

    root_path = "./data/rpe-03_w3"
    batch_size = 32
    num_workers = 16
    image_size = 256

    device = "cuda"

    dataloader = get_dataloader(root_path, batch_size=batch_size, \
            num_workers=num_workers, image_size=image_size)


    num_epochs = 12000
    display_every = 10
    learning_rate = 1e-4

    deconn = Deconn()
    deconn.to(device)

    optimizer = torch.optim.Adam(deconn.parameters(), lr=learning_rate)


    try:
        for epoch in range(num_epochs):


            for ii, data in enumerate(dataloader,0):


                deconn.zero_grad()
                data_in = data[0][:,0:1,:,:].to(device)

                recon, decon = deconn(data_in)

                loss = 0.8 * torch.mean(torch.abs(recon-data_in)**2) \
                        + 0.2 * torch.mean(torch.abs(decon))

                loss.backward()

                optimizer.step()

                if epoch % display_every == 0:

                    if not(ii):
                        running_loss = 0.0

                    running_loss += loss

            if (epoch-1) % display_every == 0:
                running_loss /= ii
                print("batch loss at epoch {} =  {:.4e}".format(epoch-1, running_loss))



    except KeyboardInterrupt:

        import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()


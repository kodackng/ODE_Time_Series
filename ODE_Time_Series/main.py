import argparse
from torchdiffeq import odeint_adjoint as odeint
import torchdiffeq
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from ode_data_loader import dataloader_ode
from ode_model import f, ODEBlock, ODENet
from torch.optim.lr_scheduler import StepLR

fpath = 'where_csv_file_is_located.csv'
train_loader = DataLoader(dataloader_ode(filepath=fpath,train=True),batch_size=24, shuffle=False)
test_loader = DataLoader(dataloader_ode(filepath=fpath,train=False),batch_size=25, shuffle=False)
scale_max = 4000.0
scale_min = 1785
new_max = 1.0
new_min = -1.0

scale_maxy = 4000.0
scale_miny = 1820
new_maxy = 1.0
new_miny = -1.0

def train(args, model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        x, y = batch.to(device)
        optimizer.zero_grad()
        x_len = np.rint(len(x)/2)
        
        x_learn = x[0:int(x_len),1:]
        y_trgt = x[:,0].clone()
        future_periods = len(x) - len(x_learn)
        
        preds = model(x_learn, future_periods)
        loss = loss_func(preds.squeeze(0), y_trgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break  
            
def test(model, device, test_loader, loss_func, prediction=False):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in test_loader:
            x, y = batch.to(device)
            len_data = len(x)
            x_len = np.rint(len(x)/2)
            y = x[:,0].clone()
            if np.rint(len(x)/2) > 0:
                x_learn = x[0:int(x_len), 1:]
            else:
                x_learn = x[0:,1:]
            
            future_periods = x_len = len(x_learn)
            
            if prediction == True:
                future_periods = 13
                x_learn = x[0:,1:]
            
            outputs = model(x_learn, future_periods)
            loss = loss_func(outputs.squeeze(0)[0:int(len(y))], y)
            test_loss += loss.item()
            
        outputs = (outputs.squeeze() - torch.Tensor([new_min]))/(torch.Tensor([new_max-new_min])*torch.Tensor([scale_max-scale_min])+torch.Tensor([scale_min]))
        y = (y.squeeze() - torch.Tensor([new_miny]))/(torch.Tensor([new_maxy-new_miny])*torch.Tensor([scale_maxy-scale_miny])+torch.Tensor([scale_miny]))
        
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))
        print("\nPredicted Values: {:.4f}\n".format(outputs))     
        print("\nTarget Values: {:.4f}\n".format(y))
        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description = 'PyTorch ODE Time Series Example')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 24)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 25)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        
    model = ODENet(in_dim=12, mid_dim=200, out_dim=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    loss_func = nn.MSELoss(reduction='sum')
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device,train_loader,optimizer,loss_func,epoch)
        test(model, device, test_loader, loss_func, prediction=False)
        scheduler.step()
        
    if args.save_model:
        torch.save(model.state_dict(), "ode_time_series.pt")     
        
if __name__ == '__main__':
    main()
             
        
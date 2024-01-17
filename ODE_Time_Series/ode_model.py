import torch 
import torch.nn as nn
import torchdiffeq
from torchdiffeq import odeint_adjoint as odeadj

class f(nn.module):
    """_summary_

    Args:
        model (torch model): function to be passed to ODEBlock
    """
    def __init__(self, dim):
        super(f, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 124),
            nn.ReLU(),
            nn.Linear(124, 124),
            nn.ReLU(),
            nn.Linear(124, dim),
            nn.Tanh()
        )
    
    def forward(self, t, x):
        return self.model(x)

class ODEBlock(nn.Module):
    """_summary_
    This is the ODE block which acts as a sort of wrapper around the ODE Solver,
    allows an easy connection to the newtork and solver.

    Args:
        odefunc (torch model): f
        rtol (float): error tolerance
        atol (float): error tolerance
        solver (ODE solver,str): 'dopri5'
        use_adjoint (bool): 
        integration_time (float): time steps
    """
    def __init_(self, f, solver: str = 'dopri5', rtol: float = 1e-4,
                atol: float = 1e-4, adjoint: bool = True, autonomous: bool = True):
        super(ODEBlock, self).__init__()
        self.odefunc = f
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.use_adjoint = adjoint
        self.integration_time = torch.Tensor([0,1]).float()
    
    @property
    def ode_method(self):
        return torchdiffeq.odeint_adjoint if self.use_adjoint else torchdiffeq.odeint
    
    def forward(self, x: torch.Tensor, adjoint: bool = True, integration_time=None):
        integration_time = self.integration_time if integration_time is None else integration_time
        ode_method = torchdiffeq.odeint_adjoint if self.use_adjoint else torchdiffeq.odeint 
        out = ode_method(
            self.odefunc, x, integration_time, rtol=self.rtol,
            atol=self.atol, method=self.solver)
        return out

class ODENet(nn.Module):
    """_summary_

    Args:
        f (torch model): torch model passed to ODE wrapper
        ode_block (torch model, ODE Solver): ODEBlock
    """
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ODENet, self).__init__()
        fx = f(dim=mid_dim)
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(mid_dim)
        self.ode_block = ODEBlock(fx)
        self.dropout = nn.Dropout(0.4)
        self.norm2 = nn.LayerNorm(mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        
    def forward(self, x, future_preds=1):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        outputs = []
        
        for time_step in x.split(1,dim=0):
            out = self.fc1(time_step)
            out = self.relu(out)
            out = self.norm1(out)
            out = self.ode_block(out)
            out = self.norm2(out)
            out = self.dropout(out)
            output = torch.mean(self.fc2(out),dim=0)
            outputs.append(output)
            
        lags = time_step
        for step in range(future_preds):
            lags = torch.concat([output, lags[:,0:-1]],dim=1)
            out = self.fc1(lags)
            out = self.relu(out)
            out = self.norm1(out)
            out = self.ode_block(out)
            out = self.norm2(out)
            out = self.dropout(out)
            output = torch.mean(self.fc2(out),dim=0)
            outputs.append(output)
        outputs = torch.cat(outputs,dim=1)
        return outputs


        
        
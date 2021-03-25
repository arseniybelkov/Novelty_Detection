import torch
import torch.nn.functional as F


def train_model(r_net: torch.nn.Module,
				d_net: torch.nn.Module,
                train_dataset: torch.utils.data.Dataset,
                valid_dataset: torch.utils.data.Dataset,
                loss_function: torch.nn.Module,
                optim_params_r: Dict = {},
                optim_params_d: Dict = {},
                initial_lr = 0.001,
                batch_size = 512,
                max_epochs = 1000, 
                best_model_root = './best_model.pth'):

	optim_r = torch.optim.Adam(r_net.parameters(), lr = initial_lr, **optim_r_params)
	optim_d = torch.optim.Adam(d_net.parameters(), lr = initial_lr, **optim_d_params)

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    for epoch in range(max_epochs):

    	d_loss = train_dnet(d_net, optim_d, loss_function, train_loader)
    	r_loss = train_rnet(r_net, optim_r, loss_function, train_loader)
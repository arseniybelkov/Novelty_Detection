import torch
import torch.nn.functional as F


def train_model(r_net: torch.nn.Module,
				d_net: torch.nn.Module,
				train_dataset: torch.utils.data.Dataset,
				valid_dataset: torch.utils.data.Dataset,
				loss_function: torch.nn.Module,
				optim_r_params: dict = {},
				optim_d_params: dict = {},
				learning_rate: float = 0.001,
				batch_size: int = 512,
				max_epochs: int = 1000,
				rec_loss_bound: float = 0.001,
				device: torch.device = torch.device('cpu'),
				best_model_root: str = './model.pth'):

	optim_r = torch.optim.Adam(r_net.parameters(), lr = learning_rate, **optim_r_params)
	optim_d = torch.optim.Adam(d_net.parameters(), lr = learning_rate, **optim_d_params)

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

	metrics = {'train_loss' : [], 'valid_loss' : []}

	for epoch in range(max_epochs):

		train_loss = train_single_epoch(r_net, d_net, optim_r, optim_d, loss_function, train_loader)

		valid_loss = validate_single_epoch(d_net, r_net, loss_function, valid_loader)

		metrics['train_loss'].append(train_loss)
		metrics['valid_loss'].append(valid_loss)

		if r_loss <= rec_loss_bound:
			print('Reconstruction loss achieved optimum\n \
				Stopping training')
			break

	plot_learning_curves(metrics)


def train_single_epoch(r_net, d_net, optim_r, optim_d, loss_function, train_loader):

	r_net.train()
	d_net.train()
	total_loss = []

	for data in train_loader:

		# Train on real data
		x_real = data[0].to(device)
		y_real = torch.ones_like(x_real)

		pred_real = d_net(x_real)

		real_loss = loss_function(pred_real, y_real, x_real, x_real)
		real_loss.backward()

		# Train on fake data
		x_fake = r_net(x_real)
		y_fake = torch.zeros_like(x_fake)

		pred_fake = d_net(x_fake)

		fake_loss = loss_function(pred_fake, y_fake, x_real, x_fake)
		fake_loss.backward()

		total_loss.append(real_loss + fake_loss)

		optim_d.step()
		optim_r.step()

	avg_train_loss = sum(total_loss) / len(total_loss)

	return avg_train_loss


def validate_single_epoch(r_net, d_net, loss_fuctnion, valid_loader):

	r_net.eval()
	d_net.eval()

	total_loss = []

	for data in valid_loader:

		x_real = data[0].to(device)
		y_real = torch.ones_like(x_real)

		x_fake = r_net(x_real)
		y_fake = torch.zeros_like(x_fake)

		pred_real = d_net(x_real)
		pred_fake = d_net(x_fake)

		real_loss = loss_function(pred_real, y_real, x_real, x_real)
		fake_loss = loss_function(pred_fake, y_fake, x_real, x_fake)

		total_loss.append(real_loss + fake_loss)

	avg_valid_loss = sum(total_loss) / len(total_loss)

	return avg_valid_loss


def plot_learning_curves(metrics):

	pass #tu tu ru ...
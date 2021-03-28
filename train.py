import torch
import matplotlib.pyplot as plt


def train_model(r_net: torch.nn.Module,
				d_net: torch.nn.Module,
				train_dataset: torch.utils.data.Dataset,
				valid_dataset: torch.utils.data.Dataset,
				r_loss,
				d_loss,
				optim_r_params: dict = {},
				optim_d_params: dict = {},
				learning_rate: float = 0.001,
				batch_size: int = 512,
				max_epochs: int = 1000,
				rec_loss_bound: float = 0.001,
				lambd: float = 0.4,
				device: torch.device = torch.device('cpu')) -> tuple:

	optim_r = torch.optim.Adam(r_net.parameters(), lr = learning_rate, **optim_r_params)
	optim_d = torch.optim.Adam(d_net.parameters(), lr = learning_rate, **optim_d_params)

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers = 2)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

	metrics =  {'train' : {'rec_loss' : [], 'gen_loss' : [], 'dis_loss' : []},
				'valid' : {'rec_loss' : [], 'gen_loss' : [], 'dis_loss' : []}}

	for epoch in range(max_epochs):

		train_metrics = train_single_epoch(r_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device)
		valid_metrics = validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device)

		metrics['train']['rec_loss'].append(train_metrics['rec_loss'].item())
		metrics['train']['gen_loss'].append(train_metrics['gen_loss'].item())
		metrics['train']['dis_loss'].append(train_metrics['dis_loss'].item())
		metrics['valid']['rec_loss'].append(valid_metrics['rec_loss'].iten())
		metrics['valid']['gen_loss'].append(valid_metrics['gen_loss'].item())
		metrics['valid']['dis_loss'].append(valid_metrics['dis_loss'].item())

		if epoch % 10 == 0:

			print(f'Epoch {epoch}:')
			print('TRAIN METRICS:', avg_train_metrics)
			print('VALID METRICS:', avg_valid_metrics)

			torch.save(r_net, './r_net.pth')
			torch.save(d_net, './d_net.pth')

		if avg_valid_metrics['rec_loss'].item() < rec_loss_bound:
			print('Reconstruction loss achieved optimum\n \
				Stopping training')

			torch.save(r_net, './r_net.pth')
			torch.save(d_net, './d_net.pth')

			break

	plot_learning_curves(metrics)

	return (r_net, d_net)


def train_single_epoch(r_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device) -> dict:

	r_net.train()
	d_net.train()

	train_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'dis_loss' : 0}

	for data in train_loader:

		x_real = data[0].to(device)
		x_fake = r_net(x_real)

		d_net.zero_grad()

		L_rd = d_loss(d_net, x_real, x_fake)

		L_rd.backward()
		optim_d.step()

		r_net.zero_grad()

		r_metrics, L_r = r_loss(d_net, x_real, x_fake, lambd)

		L_r.backward()
		optim_r.step()

		train_metrics['rec_loss'] += r_metrics['rec_loss']
		train_metrics['gen_loss'] += r_metrics['gen_loss']
		train_metrics['dis_loss'] += L_rd

	train_metrics['rec_loss'] = train_metrics['rec_loss'] / len(train_loader.dataset)
	train_metrics['gen_loss'] = train_metrics['gen_loss'] / len(train_loader.dataset)
	train_metrics['dis_loss'] = train_metrics['dis_loss'] / len(train_loader.dataset)

	return train_metrics


def validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device) -> dict:

	r_net.eval()
	d_net.eval()

	valid_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'dis_loss' : 0}

	for data in valid_loader:

		x_real = data[0].to(device)
		x_fake = r_net(x_real)

		loss_rd = d_loss(d_net, x_real, x_fake)

		r_metrics, _ = r_loss(d_net, x_real, x_fake, 0)
		
		valid_metrics['rec_loss'] += r_metrics['rec_loss']
		valid_metrics['gen_loss'] += r_metrics['gen_loss']
		valid_metrics['dis_loss'] += loss_rd

	valid_metrics['rec_loss'] = valid_metrics['rec_loss'] / len(train_loader.dataset)
	valid_metrics['gen_loss'] = valid_metrics['gen_loss'] / len(train_loader.dataset)
	valid_metrics['dis_loss'] = valid_metrics['dis_loss'] / len(train_loader.dataset)

	return valid_metrics


def plot_learning_curves(metrics: dict):

	# Plot reconstruction loss: ||X' - X||^2
	plt.plot(metrics['train']['rec_loss'], label = 'Train rec loss')
	plt.plot(metrics['valid']['rec_loss'], label = 'Dev rec loss')
	plt.title('Reconstruction loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Rec loss')
	plt.legend()
	plt.savefig('./metrics/rec_loss.jpg')

	# Plot discriminator loss: -y*log(D(x)) - (1-y)*(1 - D(R(x)))
	plt.plot(metrics['train']['dis_loss'], label = 'Train dis loss')
	plt.plot(metrics['valid']['dis_loss'], label = 'Dev dis loss')
	plt.title('Discriminator loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Dis loss')
	plt.legend()
	plt.savefig('./metrics/dis_loss.jpg')

	# Plot generator loss: -log(D(R(x)))
	plt.plot(metrics['train']['gen_loss'], label = 'Train gen loss')
	plt.plot(metrics['valid']['gen_loss'], label = 'Dev gen loss')
	plt.title('Generator loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Gen loss')
	plt.legend()
	plt.savefig('./metrics/gen_loss.jpg')
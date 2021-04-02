import os
import torch
import matplotlib.pyplot as plt


def train_model(r_net: torch.nn.Module,
				d_net: torch.nn.Module,
				train_dataset: torch.utils.data.Dataset,
				valid_dataset: torch.utils.data.Dataset,
				r_loss,
				d_loss,
				lr_scheduler = None,
				optim_r_params: dict = {},
				optim_d_params: dict = {},
				learning_rate: float = 0.001,
				scheduler_r_params: dict = {},
				scheduler_d_params: dict = {},
				batch_size: int = 512,
				pin_memory: bool = True,
				num_workers: int = 1,
				max_epochs: int = 1000,
				epoch_step: int = 1,
				save_step: int = 5,
				rec_loss_bound: float = 0.01,
				lambd: float = 0.4,
				device: torch.device = torch.device('cpu'),
				save_path: tuple = ('.','r_net.pth','d_net.pth')) -> tuple:

	model_path = os.path.join(save_path[0], 'models')
	metric_path= os.path.join(save_path[0], 'metrics')
	r_net_path = os.path.join(model_path, save_path[1])
	d_net_path = os.path.join(model_path, save_path[2])

	if not os.path.exists(model_path):
		os.makedirs(model_path)
	if not os.path.exists(metric_path):
		os.makedirs(metric_path)

	print(f'Models will be saved in {model_path}')
	print(f'Metrics will be saved in {metric_path}')

	optim_r = torch.optim.Adam(r_net.parameters(), lr = learning_rate, **optim_r_params)
	optim_d = torch.optim.Adam(d_net.parameters(), lr = learning_rate, **optim_d_params)

	if lr_scheduler:
		scheduler_r = lr_scheduler(optim_r, verbose=True, **scheduler_r_params)
		scheduler_d = lr_scheduler(optim_d, verbose=True, **scheduler_d_params)

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

	metrics =  {'train' : {'rec_loss' : [], 'gen_loss' : [], 'dis_loss' : []},
				'valid' : {'rec_loss' : [], 'gen_loss' : [], 'dis_loss' : []}}

	for epoch in range(max_epochs):

		train_metrics = train_single_epoch(r_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device)
		valid_metrics = validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device)

		metrics['train']['rec_loss'].append(train_metrics['rec_loss'])
		metrics['train']['gen_loss'].append(train_metrics['gen_loss'])
		metrics['train']['dis_loss'].append(train_metrics['dis_loss'])
		metrics['valid']['rec_loss'].append(valid_metrics['rec_loss'])
		metrics['valid']['gen_loss'].append(valid_metrics['gen_loss'])
		metrics['valid']['dis_loss'].append(valid_metrics['dis_loss'])

		if epoch % epoch_step == 0:
			print(f'Epoch {epoch}:')
			print('TRAIN METRICS:', train_metrics)
			print('VALID METRICS:', valid_metrics)

		if lr_scheduler:
			scheduler_r.step(valid_metrics['rec_loss'])
			scheduler_d.step(valid_metrics['rec_loss'])

		if epoch % save_step == 0:
			torch.save(r_net, r_net_path)
			torch.save(d_net, d_net_path)
			print(f'Saving model on epoch {epoch}')

		if valid_metrics['rec_loss'] < rec_loss_bound and train_metrics['rec_loss'] < rec_loss_bound:
			torch.save(r_net, r_net_path)
			torch.save(d_net, d_net_path)
			print('Reconstruction loss achieved optimum')
			print('Stopping training')

			break

	plot_learning_curves(metrics, metric_path)

	return (r_net, d_net)


def train_single_epoch(r_net, d_net, optim_r, optim_d, r_loss, d_loss, train_loader, lambd, device) -> dict:

	r_net.train()
	d_net.train()

	train_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'dis_loss' : 0}

	for data in train_loader:

		x_real = data[0].to(device)
		x_fake = r_net(x_real)

		d_net.zero_grad()

		dis_loss = d_loss(d_net, x_real, x_fake)

		dis_loss.backward()
		optim_d.step()

		r_net.zero_grad()

		r_metrics = r_loss(d_net, x_real, x_fake, lambd) # L_r = gen_loss + lambda * rec_loss

		r_metrics['L_r'].backward()
		optim_r.step()

		train_metrics['rec_loss'] += r_metrics['rec_loss']
		train_metrics['gen_loss'] += r_metrics['gen_loss']
		train_metrics['dis_loss'] += dis_loss

	train_metrics['rec_loss'] = train_metrics['rec_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)
	train_metrics['gen_loss'] = train_metrics['gen_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)
	train_metrics['dis_loss'] = train_metrics['dis_loss'].item() / (len(train_loader.dataset) / train_loader.batch_size)

	return train_metrics


def validate_single_epoch(r_net, d_net, r_loss, d_loss, valid_loader, device) -> dict:

	r_net.eval()
	d_net.eval()

	valid_metrics = {'rec_loss' : 0, 'gen_loss' : 0, 'dis_loss' : 0}

	with torch.no_grad():
		for data in valid_loader:

			x_real = data[0].to(device)
			x_fake = r_net(x_real)
	
			dis_loss = d_loss(d_net, x_real, x_fake)
	
			r_metrics = r_loss(d_net, x_real, x_fake, 0)
				
			valid_metrics['rec_loss'] += r_metrics['rec_loss']
			valid_metrics['gen_loss'] += r_metrics['gen_loss']
			valid_metrics['dis_loss'] += dis_loss

	valid_metrics['rec_loss'] = valid_metrics['rec_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
	valid_metrics['gen_loss'] = valid_metrics['gen_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)
	valid_metrics['dis_loss'] = valid_metrics['dis_loss'].item() / (len(valid_loader.dataset) / valid_loader.batch_size)

	return valid_metrics


def plot_learning_curves(metrics: dict, metric_path: str):

	rec_loss_path = os.path.join(metric_path, 'rec_loss.jpg')
	dis_loss_path = os.path.join(metric_path, 'dis_loss.jpg')
	gen_loss_path = os.path.join(metric_path, 'gen_loss.jpg')

	# Plot reconstruction loss: ||X' - X||^2
	plt.plot(metrics['train']['rec_loss'], label = 'Train rec loss')
	plt.plot(metrics['valid']['rec_loss'], label = 'Dev rec loss')
	plt.title('Reconstruction loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Rec loss')
	plt.legend()
	plt.savefig(rec_loss_path)

	# Plot discriminator loss: -log(D(x)) - log(1 - D(R(x')))
	plt.plot(metrics['train']['dis_loss'], label = 'Train dis loss')
	plt.plot(metrics['valid']['dis_loss'], label = 'Dev dis loss')
	plt.title('Discriminator loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Dis loss')
	plt.legend()
	plt.savefig(dis_loss_path)

	# Plot generator loss: -log(D(R(x)))
	plt.plot(metrics['train']['gen_loss'], label = 'Train gen loss')
	plt.plot(metrics['valid']['gen_loss'], label = 'Dev gen loss')
	plt.title('Generator loss evolution')
	plt.xlabel('epochs')
	plt.ylabel('Gen loss')
	plt.legend()
	plt.savefig(gen_loss_path)
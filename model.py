import torch
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):

	def __init__(self, dataset: torch.utils.data.Dataset, labels: list):
		self.dataset = dataset
		self.labels = labels
		self.indexes = self._extract_indexes()

	def _extract_indexes(self):

		indexes = []

		for label in self.labels:
			for i, sample in enumerate(self.dataset):
				if sample[1] == label:
					indexes.append(i)
		return indexes

	def __len__(self):
		return len(self.indexes)
	
	def __getitem__(self, idx):
		return self.dataset[self.indexes[idx]]


class R_Net(torch.nn.Module):

	def __init__(self, activation = torch.nn.SELU, in_channels:int = 3, n_channels:int = 64, kernel_size:int = 5, std = 1):

		super(R_Net, self).__init__()

		self.activation = activation
		self.in_channels = in_channels
		self.n_c = n_channels
		self.k_size = kernel_size
		self.std = std

		self.Encoder = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c),
											self.activation(),
											torch.nn.Conv2d(self.n_c, self.n_c*2, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*2),
											self.activation(),
											torch.nn.Conv2d(self.n_c*2, self.n_c*4, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*4),
											self.activation(),
											torch.nn.Conv2d(self.n_c*4, self.n_c*8, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*8),
											self.activation())

		self.Decoder = torch.nn.Sequential(torch.nn.ConvTranspose2d(self.n_c*8, self.n_c*4, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*4),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c*4, self.n_c*2, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*2),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c*2, self.n_c, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c, self.in_channels, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.in_channels),
											self.activation())

	def forward(self, x):

		x_hat = self.add_noise(x)
		z = self.Encoder(x_hat)
		x_out = self.Decoder(z)

		return x_out

	def add_noise(self, x):

		noise = torch.randn_like(x) * self.std
		x_hat = x + noise

		return x_hat


class D_Net(torch.nn.Module):

	def __init__(self, in_resolution:tuple, activation = torch.nn.SELU, in_channels:int = 3, n_channels:int = 64, kernel_size:int = 5):

		super(D_Net, self).__init__()

		self.activation = activation
		self.in_resolution = in_resolution
		self.in_channels = in_channels
		self.n_c = n_channels
		self.k_size = kernel_size

		self.cnn = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c),
										self.activation(),
										torch.nn.Conv2d(self.n_c, self.n_c*2, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*2),
										self.activation(),
										torch.nn.Conv2d(self.n_c*2, self.n_c*4, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*4),
										self.activation(),
										torch.nn.Conv2d(self.n_c*4, self.n_c*8, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*8),
										self.activation())

		# Compute output dimension after conv part of D network

		self.out_dim = self._compute_out_dim()

		self.fc = torch.nn.Sequential(torch.nn.Linear(self.out_dim, 1),
										torch.nn.Sigmoid())

	def _compute_out_dim(self):
		
		test_x = torch.Tensor(1, self.in_channels, self.in_resolution[0], self.in_resolution[1])
		for p in self.cnn.parameters():
			p.requires_grad = False
		test_x = self.cnn(test_x)
		out_dim = torch.prod(torch.tensor(test_x.shape[1:])).item()
		for p in self.cnn.parameters():
			p.requires_grad = True

		return out_dim

	def forward(self, x):

		x = self.cnn(x)

		x = torch.flatten(x, start_dim = 1)

		out = self.fc(x)

		return out


def R_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

	pred = d_net(x_fake)
	y = torch.ones_like(pred)

	rec_loss = F.mse_loss(x_fake, x_real, reduction = 'sum')
	gen_loss = F.binary_cross_entropy(pred, y, reduction = 'sum') # generator loss

	L_r = gen_loss + lambd * rec_loss

	return {'rec_loss' : rec_loss, 'gen_loss' : gen_loss}, L_r


def D_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:

	pred_real = d_net(x_real)
	pred_fake = d_net(x_fake.detach())

	y_real = torch.ones_like(pred_real)
	y_fake = torch.ones_like(pred_fake)

	real_loss = F.binary_cross_entropy(pred_real, y_real, reduction = 'sum')
	fake_loss = F.binary_cross_entropy(pred_fake, y_fake, reduction = 'sum')

	return real_loss + fake_loss
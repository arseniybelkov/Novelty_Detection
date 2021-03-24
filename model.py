import torch


class R_Net(torch.nn.Module):

	def __init__(self, in_channels = 3, n_channels = 64, kernel_size = 5, std = 1):

		super(R_Net, self).__init__()

		self.in_channels = in_channels
		self.n_c = n_channels
		self.k_size = kernel_size
		self.std = std

		self.enconv1 = torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size)
		self.en_bn1 = torch.nn.BatchNorm2d(self.n_c)
		self.enconv2 = torch.nn.Conv2d(self.n_c, self.n_c*2, self.k_size)
		self.en_bn2 = torch.nn.BatchNorm2d(self.n_c*2)
		self.enconv3 = torch.nn.Conv2d(self.n_c*2, self.n_c*4, self.k_size)
		self.en_bn3 = torch.nn.BatchNorm2d(self.n_c*4)
		self.enconv4 = torch.nn.Conv2d(self.n_c*4, self.n_c*8, self.k_size)
		self.en_bn4 = torch.nn.BatchNorm2d(self.n_c*8)

		self.deconv1 = torch.nn.ConvTranspose2d(self.n_c*8, self.n_c*4, self.k_size)
		self.de_bn1 = torch.nn.BatchNorm2d(self.n_c*4)
		self.deconv2 = torch.nn.ConvTranspose2d(self.n_c*4, self.n_c*2, self.k_size)
		self.de_bn2 = torch.nn.BatchNorm2d(self.n_c*2)
		self.deconv3 = torch.nn.ConvTranspose2d(self.n_c*2, self.n_c, self.k_size)
		self.de_bn3 = torch.nn.BatchNorm2d(self.n_c)
		self.deconv4 = torch.nn.ConvTranspose2d(self.n_c, self.in_channels, self.k_size)

	def forward(self, x):

		x_hat = add_noise(x)
		z = encoder(x_hat)
		x_out = decoder(z)

		return x_out

	def add_noise(self, x):

		noise = torch.randn_like(x) * self.std
		x_hat = x + noise

		return x_hat

	def encoder(self, x):

		x = self.en_bn1(torch.nn.SELU(self.enconv1(x)))
		x = self.en_bn2(torch.nn.SELU(self.enconv2(x)))
		x = self.en_bn3(torch.nn.SELU(self.enconv3(x)))
		z = self.en_bn4(torch.nn.SELU(self.enconv4(x)))

		return z

	def decoder(self, z):

		z = torch.de_bn1(torch.nn.SELU(self.deconv1(x)))
		z = torch.de_bn2(torch.nn.SELU(self.deconv2(x)))
		z = torch.de_bn3(torch.nn.SELU(self.deconv3(x)))
		x_out = torch.nn.SELU(self.deconv4(x))

		return x_out


class D_Net(torch.nn.Module):

	def __init__(self, in_resolution, in_channels = 3, n_channels = 64, kernel_size = 5):

		super(D_Net, self).__init__()

		self.in_resolution = in_resolution
		self.in_channels = in_channels
		self.n_c = n_channels
		self.k_size = kernel_size

		self.conv1 = torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size)
		self.bn1 = torch.nn.BatchNorm2d(self.n_c)
		self.conv2 = torch.nn.Conv2d(self.n_c, self.n_c*2, self.k_size)
		self.bn2 = torch.nn.BatchNorm2d(self.n_c*2)
		self.conv3 = torch.nn.Conv2d(self.n_c*2, self.n_c*4, self.k_size)
		self.bn3 = torch.nn.BatchNorm2d(self.n_c*4)
		self.conv4 = torch.nn.Conv2d(self.n_c*4, self.n_c*8, self.k_size)
		self.bn4 = torch.nn.BatchNorm2d(self.n_c*8)

		# Compute output dimension after conv part of D network
		test_x = torch.Tensor(1, self.in_channels, self.in_resolution[0], self.in_resolution[1])
		out = self.cnn(test_x)
		self._out_dim =  torch.prod(torch.tensor(out.shape[1:])).item()

		self.fc1 = torch.nn.Linear(self.out_dim, 1024)
		self.bn5 = torch.nn.BatchNorm1d(1024)
		self.fc2 = torch.nn.Linear(1024, 1)

	def forward(self, x):

		x = self.cnn(x)
		x = torch.flatten(x, dim = 1)
		x = self.bn5(.torch.nn.SELU(self.fc1(x)))
		out = torch.nn.Sigmoid(self.fc2(x))

		return out

	def cnn(self, x):

		x = self.bn1(torch.nn.SELU(self.conv1(x)))
		x = self.bn2(torch.nn.SELU(self.conv2(x)))
		x = self.bn3(torch.nn.SELU(self.conv3(x)))
		x = self.bn4(torch.nn.SELU(self.conv4(x)))

		return x
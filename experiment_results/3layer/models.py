# COMP4680/8650: ADVANCED TOPICS IN STATISTICAL MACHINE LEARNING
# ASSIGNMENT 6
#
# ONLY MODIFY CODE IN THIS FILE
#

import torch
import torch.nn as nn

# TODO: Write any helper routines here.


class Encoder(nn.Module):
	"""Encoder network to map from an RGB image to a latent feature vector."""

	def __init__(self, z_dim=64, img_size=64):
		super(Encoder, self).__init__()

		self.z_dim = z_dim
		self.hidden_layer = nn.Sequential(
							  nn.Linear(img_size**2*3, z_dim),
							  nn.BatchNorm1d(z_dim),
							  nn.ReLU()
							  )
		self.output_layer = nn.Sequential(
							  nn.Linear(z_dim, z_dim),
							  nn.Tanh()
							  )

		# TODO: Create a nn.Sequential model for each layer in the encoder as
		# described in the assignment specification and assign them to
		# self.hidden_layer and self.output_layer.



	def forward(self, x):
		x = x.view(x.size()[0], -1)
		x = self.output_layer(self.hidden_layer(x))
		return x


class Decoder(nn.Module):
	"""Decoder network to map from a latent feature vector to an RGB image."""

	def __init__(self, z_dim=64, img_size=64):
		super(Decoder, self).__init__()

		assert img_size==64
		self.z_dim = z_dim
		self.layer1 = nn.Sequential(
							  nn.ConvTranspose2d(z_dim, 128, kernel_size=7, stride=1),
							  nn.BatchNorm2d(128),
							  nn.ReLU(),
							  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
							  nn.BatchNorm2d(128),
							  nn.ReLU(),
							  )
							  
		self.layer2 = nn.Sequential(
							  nn.ConvTranspose2d(128, 64, kernel_size=7, stride=4),
							  nn.BatchNorm2d(64),
							  nn.ReLU(),
							  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
							  nn.BatchNorm2d(64),
							  nn.ReLU(),
							  )
		
		
		self.layer3 = nn.Sequential(
							  nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
							  nn.BatchNorm2d(32),
							  nn.ReLU(),
							  nn.Conv2d(32, 3, kernel_size=2, stride=1, padding=1),
							  nn.Tanh(),
							  )
		
		# TODO: Create a nn.Sequential model for each layer in the decoder as
		# described in the assignment specification. Assign them to self.layer1,
		# self.layer2, etc. The test_models.py script will help you check that
		# you got the layers correct.



	def forward(self, x):
		x = x.view(x.size()[0], self.z_dim, 1, 1)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		return x

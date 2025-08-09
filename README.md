# HRM Zero for Snake

I noticed HRM is expensive if T is high so I modified it to be cheap. It still seems to work on Snake and also alleviates the issue of eventual collapse.

```py
class HRM(nn.Module):
	def __init__(self, output_size=4):
		super(HRM, self).__init__()
		self.L_net = Net()
		self.H_net = Net()
		self.output_head = OutputHead(output_size)
		self.input_embedding = ConvNet()
		self.i = 0
	def forward(self, z, x, N=2, T=10):
		x = x.to(device)
		x = self.input_embedding(x)
		zH, zL = z
		zH = zH.to(device)
		zL = zL.to(device)
		# with torch.no_grad():
		# 	for _i in range(N * T - 1):
		# 		zL = self.L_net(zL, zH, x)
		# 		if (_i + 1) % T == 0:
		# 			zH = self.H_net(zH, zL)

		self.i += 1

		with torch.no_grad():
			zL = self.L_net(zL, zH, x)
			zH = self.H_net(zH, zL, x)

		assert not zH.requires_grad and not zL.requires_grad

		# 1−step grad
		zL = self.L_net(zL, zH, x)
		if self.i % T == 0:
			zH = self.H_net(zH, zL, x)
		return (zH, zL), self.output_head(zH)
```

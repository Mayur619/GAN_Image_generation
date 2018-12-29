import torch
import torchvision
from torch.autograd import Variable

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!=-1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

batch_size=64
image_size=64
transform=torchvision.transforms.Compose([torchvision.transforms.Scale(image_size),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
dataset=torchvision.datasets.CIFAR10(root='/home/mayur/cifardata',download=True,transform=transform)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=2)

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main=torch.nn.Sequential(
                torch.nn.ConvTranspose2d(100,512,4,1,0,bias=False),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(512,256,4,2,1,bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(256,128,4,2,1,bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(128,64,4,2,1,bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(64,3,4,2,1,bias=False),
                torch.nn.Tanh()
                )
    def forward(self,input_):
        output=self.main(input_)
        return output

netG=Generator()
netG.apply(weights_init)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main=torch.nn.Sequential(
                torch.nn.Conv2d(3,64,4,2,1,bias=False),
                torch.nn.LeakyReLU(0.2,True),
                torch.nn.Conv2d(64,128,4,2,1,bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU(0.2,True),
                torch.nn.Conv2d(128,256,4,2,1,bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU(0.2,True),
                torch.nn.Conv2d(256,512,4,2,1,bias=False),
                torch.nn.BatchNorm2d(512),
                torch.nn.LeakyReLU(0.2,True),
                torch.nn.Conv2d(512,1,4,1,0,bias=False),
                torch.nn.Sigmoid()
                )
    def forward(self,input_):
        return self.main(input_).view(-1)

netD=Discriminator()
netD.apply(weights_init)

criterion=torch.nn.BCELoss()

optimizerD=torch.optim.Adam(netD.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerG=torch.optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5,0.999))

for epoch in range(25):
    for i,data in enumerate(dataloader,0):
        netD.zero_grad()
        real,_=data
        input_=Variable(real)
        target=Variable(torch.ones(input_.size()[0]))
        output=netD(input_)
        errD_real=criterion(output,target)
        
        noise=Variable(torch.randn(input_.size()[0],100,1,1))
        fake=netG(noise)
        target=Variable(torch.zeros(input_.size()[0]))
        output=netD(fake.detach())
        errD_fake=criterion(output,target)
        
        errD=errD_real+errD_fake
        errD.backward()
        optimizerD.step()
        
        netG.zero_grad()
        target=Variable(torch.ones(input_.size()[0]))
        output=netD(fake)
        errG=criterion(output,target)
        errG.backward()
        optimizerG.step()
        
        if i%100==0:
            print('[%d/%d] [%d/%d] Loss D:%.4f , Loss G:%.4f'%(epoch,25,i,len(dataloader),errD.item(),errG.item()))
            torchvision.utils.save_image(real,'./results/real_sample.png',normalize=True)
            fake=netG(noise)
            torchvision.utils.save_image(fake,'./results/fake_sample_epoch_%03d.png'%(epoch),normalize=True)
pred=netG(Variable(torch.randn(1,100,1,1)))
torchvision.utils.save_image(pred,'pred.png')
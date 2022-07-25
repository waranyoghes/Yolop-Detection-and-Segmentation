"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
CSPDarknet_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 2],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    "SPP", # To this point is CSPDarknet-53
    (512, 1, 1),
    (1024, 3, 1),
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    (128, 1, 1),
    "U",
    (256, 1, 1),
    (256, 3, 1),
]

Head_config= [ 
    (256,1,1),
    (128,1,1),
    "PAN",
    (256,1,1), 
    "P",
    (128,1,1),
    (128,3,2), 
    "PAN",
    (256,1,1),
    "P",
    (512,1,1),
    (512,3,2),
    "PAN",
    (512,1,1),
    "P",
]

Segmentation_config= [
      (512,1,1),
      "U",
      (256,1,1),
      #(512,3,1)
      "U",
      (128,1,1),
      "U",
      (64,1,1),
] 
      

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class CSPDenseBlock(nn.Module):
    def __init__(self, channels, use_dense=True, num_repeats=2,k=14):
        super().__init__()
        self.layers = nn.ModuleList()
        self.channels=channels
        self.layers.append(
                nn.Sequential(
                    CNNBlock(channels,channels//2, kernel_size=1),
                    CNNBlock(channels//2,k, kernel_size=3, padding=1),
                )
            )
        for repeat in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNNBlock(channels+(repeat+1)*k,(channels+(repeat+1)*k)//2, kernel_size=1),
                    CNNBlock((channels+(repeat+1)*k)//2, k, kernel_size=3, padding=1),
                )
            )
        self.Tlayer1=nn.Sequential(CNNBlock(channels+(num_repeats+1)*k,channels*2,kernel_size=1),CNNBlock(channels*2,channels,kernel_size=3,padding=1))
        self.Tlayer2=nn.Sequential(CNNBlock(2*channels,channels,kernel_size=1),torch.nn.AvgPool2d(kernel_size=3,padding=1,stride=1))
        self.k=k                          
        self.use_dense = use_dense
        self.num_repeats = num_repeats

    def forward(self, x):
        l=[]
        for layer in self.layers:
            if self.use_dense:
                #print(x.shape)
                l.append(x)
                x = torch.cat((*l,),dim=1)
                x = layer(x)  
                
            else:
                x = layer(x)
           
        l.append(x)
        x=torch.cat((*l,),dim=1)
        
        x=self.Tlayer1(x)
        #print(x.shape)
        x=self.Tlayer2(torch.cat((x,l[0]),dim=1))        
        return x   

class SPPBlock(nn.Module):
    def __init__(self,channels):         
        super().__init__()        
        self.SPP1=torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.SPP2=torch.nn.AvgPool2d(kernel_size=9, stride=1, padding=4) 
        self.SPP_out=CNNBlock(3*channels,channels,kernel_size=1)        

    def forward(self,x):
         
         return self.SPP_out(torch.cat((x,self.SPP2(x),self.SPP1(x)),dim=1))

class Detector_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class PANBlock(nn.Module):
     def __init__(self):
        super().__init__()
     def forward(self,x):
         return x


class YOLOP(nn.Module):
    def __init__(self , num_classes=10, num_seg=10, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_seg= num_seg
        self.layers = self._create_conv_layers()
        self.head=self._create_head_layers()
        self.segmentation=self._create_seg_layers()
    

    def forward(self, x,segmentation=False, detection=True):
        i=0
        outputs = []  # for each scale
        y=[]
        route_connections = []
        for layer in self.layers:
          
            #if isinstance(layer, ScalePrediction):
                #outputs.append(layer(x))
                #continue
            
            x = layer(x)
            if i==1:
              route_connections=[x]+route_connections
              i=0
            if isinstance(layer, CSPDenseBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, SPPBlock):
                route_connections=[x]+route_connections
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
                i=1
        
        if (segmentation==True):
            y=x
            #x=route_connections[0] 
            #print(route_connections[0].shape,route_connections[1].shape,route_connections[0].shape)
            for layer in self.segmentation:
              #print(y.shape)
              y = layer(y)  
      
        if (detection==True):  
           i=0
           #print(x.shape)
           for layer in self.head:
              if isinstance(layer, Detector_Head):
                  outputs.append(layer(x))
                  continue
              #i=i+1
              #print(i,layer)
              x = layer(x)
 
              if isinstance(layer, PANBlock):
                  #route_connections.pop()
                  #print(x.shape)
                  #print(route_connections[0].shape)
                  #print(route)
                  x = torch.cat([x, route_connections[0]], dim=1)
                  route_connections=route_connections[1:]
                                      
        return (outputs,y)

  
    def _create_head_layers(self):
       
       layers=nn.ModuleList()
       in_channels=self.in_channels
       #print(in_channels,"***********************crete head layer")
       for module in Head_config:
              if isinstance(module,tuple):
                 #print(in_channels)
                 out_channels, kernel_size, stride = module
                 layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                      )
                 )
                 in_channels = out_channels

              elif isinstance(module, str):
                  if module=="PAN":   
                        layers.append(PANBlock())
                        in_channels = in_channels*3
                  elif module == "P":
                      layers += [
                          ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                          CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                          Detector_Head(in_channels // 2, num_classes=self.num_classes),]
                      in_channels = in_channels // 2


       return layers


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in CSPDarknet_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        self.in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                self.in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(CSPDenseBlock(self.in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
               if module=="SPP":   
                      layers.append(SPPBlock(self.in_channels))
            
               elif module == "U":
                      layers.append(nn.Upsample(scale_factor=2),)
                      self.in_channels = self.in_channels * 3

  
        return layers
      
    def _create_seg_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        #print(in_channels,"***********************crete seg layer")
        for module in Segmentation_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, str):
               if module == "U":
                      layers.append(nn.Upsample(scale_factor=2),)
                      #self.in_channels = self.in_channels
          
        layers.append(
                        CNNBlock(
                        in_channels,
                        self.num_seg,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
        return layers


from model.layer import  *


class gtnet(nn.Module):
    def __init__(self, gcn_true, num_nodes, device, static_feat=None, dropout=0.3, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=5, in_dim=1, out_dim=2, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.seq_length=seq_length
        self.num_nodes = num_nodes
        self.gcn_depth=1
        self.dropout = dropout
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()

        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))


        kernel_size = 3
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1  #3*(7-1)+1  这是一维卷积  1*7

        if self.gcn_true:
            self.gconv1.append(mixprop(in_dim, in_dim, self.gcn_depth, dropout, propalpha))

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1   #1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)  #3

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))



                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)




    def forward(self, input_x):
        seq_len = input_x.size(3)


        # if self.gcn_true:
        #     input_x = self.gconv1[0](input_x, input_g)  # 64 43 207 13
        #5 64 1000 1
        if self.seq_length<self.receptive_field:
            input_x = nn.functional.pad(input_x,(self.receptive_field-self.seq_length,0,0,0))



        x = self.start_conv(input_x)  #5 32 1000 7  input是64*2*207*19  把2个特征值嵌入到32维里面，放到嵌入空间
        skip = self.skip0(F.dropout(input_x, self.dropout, training=self.training))


        for i in range(self.layers):

            filter = self.filter_convs[i](x) #5*32*1000*1
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x) #64 32 1000 1
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x #5 30 1000 5
            s = self.skip_convs[i](s)
            skip = s + skip


  #只取最后维度上的最后13个元素  6个都是填充的
            # if idx is None:
            #     x = self.norm[i](x,self.idx)
            # else:
            #     x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip  #64 64 207 1
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = F.relu(self.end_conv_2(x))   #64 12 207 1
        return x

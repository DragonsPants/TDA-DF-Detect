import torch;
import torch.nn as nn;
from torch.utils.data import DataLoader
from CORE import xception
from tqdm import tqdm
import torch.optim as optim

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
# model.fc? = nn.Linear( model.fc.in_features + additional features , num_classes)

class TopoHead(nn.Module):
    def __init__ (self, num_topo_features = 0, num_classes = 2):
        super(TopoHead, self).__init__()
        self.xNet = xception.xception(pretrained=True)
        for param in self.xNet.parameters():
            param.requires_grad = False

        self.land_conv1 = nn.Conv1d(2, 32, kernel_size=1)
        self.land_bn1 = nn.BatchNorm1d(32)
        self.land_conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.land_bn2 = nn.BatchNorm1d(64)
        self.land_conv3 = nn.Conv1d(64,128, kernel_size=5)
        self.land_bn3 = nn.BatchNorm1d(128)
        self.land_fc = nn.Linear( 12032, 2560 )
        self.land_fc2 = nn.Linear( 2560 , 512 )

        self.feature_conv1 = nn.Conv1d(1, 32, kernel_size=1)
        self.feature_bn1 = nn.BatchNorm1d(32)
        self.feature_conv3 = nn.Conv1d(32,128, kernel_size=3)
        self.feature_bn3 = nn.BatchNorm1d(128)
        self.feature_conv4 = nn.Conv1d(128, 64, kernel_size=5)
        self.feature_bn4 = nn.BatchNorm1d(64)
        self.feature_conv5 = nn.Conv1d(64, 32, kernel_size=7)
        self.feature_bn5 = nn.BatchNorm1d(32)
        #self.dropout = nn.Dropout(0.3)
        self.feature_fc = nn.Linear( 5760 , 512 )
        
        self.topo_fc = nn.Linear( 1024 , 1024 )
        self.fc = nn.Linear( self.xNet.fc.in_features + 1024, num_classes)

        self.relu = nn.LeakyReLU()

    def forward(self, x, gt, land):
        gt = self.relu(self.feature_bn1( self.feature_conv1(gt) ))
        gt = self.relu(self.feature_bn3( self.feature_conv3(gt) ))
        gt = self.relu(self.feature_bn4( self.feature_conv4(gt) ))
        gt = self.relu(self.feature_bn5( self.feature_conv5(gt) ))
        gt = torch.flatten(gt, start_dim=1)
        gt = self.relu(self.feature_fc( gt ))

        land = self.relu(self.land_bn1( self.land_conv1(land) ))
        land = self.relu(self.land_bn2( self.land_conv2(land) ))
        land = self.relu(self.land_bn3( self.land_conv3(land) ))
        land = torch.flatten(land, start_dim=1)
        land = self.relu(self.land_fc(land))
        land = self.relu(self.land_fc2(land) )

        t = torch.cat( (gt,land), dim = 1 )
        t = self.topo_fc(t)

        x, y = self.xNet(x)
        x = torch.cat( (x,t), dim = 1 )
        x = self.fc( x )
        return x

class xNetHead(nn.Module):
    def __init__ (self, num_topo_features = 0, num_classes = 2):
        super(xNetHead, self).__init__()
        self.xNet = xception.xception(pretrained=True)
        for param in self.xNet.parameters():
            param.requires_grad = False
        self.linear = nn.Linear( self.xNet.fc.in_features, num_classes )

    def forward(self, x, t, extra=None):
        x, y = self.xNet(x)
        x = self.linear(x)
        return x

def train_model(model, train_dataloader:DataLoader, device, num_epochs = 20):
    train_losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Running Model for {num_epochs} epochs")
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for image_batch, label_batch, topo_batch, land_batch in train_dataloader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            topo_batch = topo_batch.to(device)
            land_batch = land_batch.to(device)

            optimizer.zero_grad()
            output_batch = model( image_batch, topo_batch, land_batch )

            seg_loss = criterion( output_batch, label_batch )
            seg_loss.backward()
            optimizer.step()
            running_loss += seg_loss.item() * image_batch.size(0)
        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        print( f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}" )
    print( f"maximum training losses = {sorted(train_losses,reverse=True)[:3]}" )
    return model

def test_model( model: nn.Module, loader:DataLoader, device, num_batches = -1 ):
    num_true_pos = 0
    num_true_neg = 0
    num_false_pos = 0
    num_false_neg = 0

    model.eval()
    with torch.no_grad():
        total_correct = 0.0
        for image_batch, label_batch, grad_topo_batch, land_batch in loader:

            image_batch = image_batch.to(device)
            grad_topo_batch = grad_topo_batch.to(device)
            land_batch = land_batch.to(device)

            output = model(image_batch, grad_topo_batch, land_batch)

            #print(f"base Xnet probabilities {x}")
            #print(f"gradient topo probabilities {gxt}")

            output = torch.nn.functional.softmax(output, dim=1)
            output.cpu().numpy()

            for i in range(len(output)):
                label_guess = 0 if output[i][0] > output[i][1] else 1
                true_label = label_batch[i]

                if true_label == label_guess == 0:
                    num_true_neg += 1
                elif true_label == label_guess == 1:
                    num_true_pos += 1
                elif true_label == 0 and label_guess == 1:
                    num_false_pos += 1
                elif true_label == 1 and label_guess == 0:
                    num_false_neg += 1
                
                if label_guess == true_label:
                    total_correct += 1

    acc, prec, sens, = total_correct * 100 / len(loader.dataset), num_true_pos * 100 / ( num_false_pos + num_true_pos ), num_true_pos * 100 / ( num_false_neg + num_true_pos )

    print(f"Accuracy = {acc} %")
    print(f"Precision = {prec} %")
    print(f"Sensitivity = {sens} %")

    print(f"True Pos = { num_true_pos }, True Negative = {num_true_neg}")
    print(f"False Negative = {num_false_neg}, False Pos = { num_false_pos }")

    with open('out.txt', 'w') as output:
        output.write( f"Accuracy = {acc}, Precision = {prec}, Sensitivity = {sens}, TP = {num_true_pos}, TN = {num_true_neg}, FN = {num_false_neg}, FP = {num_false_pos}" )
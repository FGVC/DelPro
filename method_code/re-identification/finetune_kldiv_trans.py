from __future__ import print_function, division

import torch.optim as optim
from torchvision import transforms, models
import time, copy
from helper_funcs import *
import argparse
from losses import *
from trans_spline import RandomSpatialTransform
from tensorboard_logger import configure, log_value


parser = argparse.ArgumentParser()

# add arguments
# add arguments
parser.add_argument('--tr_batch_size', type=int, default=8, help="batch size for training")
parser.add_argument('--te_batch_size', type=int, default=4, help="batch size for test")
parser.add_argument('--init_lr', type=float, default=1e-3, help="starting learning rate")
parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
parser.add_argument('--mtype',type=str,default='ds121',help="Model to use |Res-18|Res-50|Des-121")
parser.add_argument('--margin', type=float, default=2.0, help="Margin for KLDiv")

FLAGS = parser.parse_args()

torch.manual_seed(42)

is_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_cuda else "cpu")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop((224,224)),
        transforms.RandomAffine(degrees=10),
        #RandomSpatialTransform(size=(224,224)),
        transforms.ColorJitter(0.05, 0.05),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Then, you can just use prebuilt torch's data loader.
data_split = np.load('data/data_split.npz')
tr_dict = {}
tr_dict['data'] = data_split['xtr']
tr_dict['labels'] = data_split['ytr']
paired_data = data_Pairedwithfew('data/train',tr_dict, data_transforms['train'])
tr_loader = torch.utils.data.DataLoader(dataset=paired_data,
                                           batch_size=FLAGS.tr_batch_size,
                                           num_workers=1, shuffle=True, drop_last=True)
datas_split = np.load('data/data_split.npz')
tedata = CustomDataset('data/train',datas_split['xte'],datas_split['yte'],transform=data_transforms['val'])
te_loader = torch.utils.data.DataLoader(dataset=tedata,
    batch_size=FLAGS.te_batch_size,
    num_workers=1)
def test(model):
    model.eval()
    num_correct = 0
    num_total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in te_loader:

            inputs, labels = inputs.to(device),labels.to(device)
            # Prediction.
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            test_loss += criterion(outputs, labels).item()
            num_total += outputs.size(0)
            num_correct += torch.sum(preds == labels.data).item()

    return test_loss/len(te_loader), (100.0 * num_correct) / num_total

def train_model(model,criterion, optimizer, lr_scheduler, num_epochs):

    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):

        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        since2 = time.time()

        lr_scheduler.step(epoch-1)
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        tot = 0.0
        cnt = 0
        # Iterate over data.
        for inputs1, inputs2, labels in tr_loader:
            inputs1, inputs2,  labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            all_inputs = torch.cat((inputs1,inputs2))
            all_labels = torch.cat((labels, labels))
            # forward

            outputs = model(all_inputs)
            preds = outputs.max(1, keepdim=True)[1]

            # cor_idx = (preds == all_labels.view_as(preds)).cpu().numpy()
            # simList, dissimList = createLabelInfoList(all_labels, cor_idx)
            simList, dissimList = createLabelInfoListng(all_labels)

            if len(simList):
                pS, qS = outputs[simList[:, 0]], outputs[simList[:, 1]]
                lossS_lab = simloss(pS, qS)
            else:
                lossS_lab = 0.0

            if len(dissimList):
                pD, qD = outputs[dissimList[:, 0]], outputs[dissimList[:, 1]]
                lossD_lab = dsimloss(pD, qD)
            else:
                lossD_lab = 0.0

            loss = criterion(outputs, all_labels) + lossS_lab + lossD_lab

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += preds.eq(all_labels.view_as(preds)).sum().item()
            tot += len(all_labels)

            if cnt % 50 == 0:
                print('[%d, %5d] loss: %.5f, Acc: %.2f' %
                      (epoch, cnt + 1, loss.item(), (100.0 * running_corrects) / tot))

            cnt = cnt + 1

        train_loss = running_loss / len(tr_loader)
        train_acc = running_corrects * 1.0 / tot

        print('Training Loss: {:.6f} Acc: {:.2f}'.format(train_loss, 100.0 * train_acc))

        test_loss, test_acc = test(model)

        print('Epoch: {:d}, Test Loss: {:.4f}, Test Acc: {:.2f}'.format(epoch, test_loss,test_acc))

        # deep copy the model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)

    time_elapsed2 = time.time() - since2
    print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed2 // 60, time_elapsed2 % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:2f}'.format(best_acc))
    return best_model

mtype = FLAGS.mtype

if mtype == 'res18':
    model_ft = models.resnet18(pretrained=True)
    model_ft.fc = nn.Linear(model_ft.fc.in_features, len(np.unique(data_split['ytr'])))
elif mtype == 'res50':
    model_ft = models.resnet50(pretrained=True)
    model_ft.fc = nn.Linear(model_ft.fc.in_features, len(np.unique(data_split['ytr'])))
elif mtype == 'ds121':
    model_ft = models.densenet121(pretrained=True)
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features, len(np.unique(data_split['ytr'])))

print( len(np.unique(data_split['ytr'])))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

simloss = KLDSimLoss()
dsimloss = KLDDsimLoss(device, margin=FLAGS.margin)

init_lr = FLAGS.init_lr
optimizer_ft = optim.SGD(model_ft.parameters(), init_lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft, [10, 15], gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler,num_epochs=FLAGS.epochs)
torch.save(model_ft.state_dict(), 'kldiv_dsnet_121_m2_joeg20epoch_correct_holdout.pt')
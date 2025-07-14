import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               optimizer:torch.optim,
               loss_fn:torch.nn.Module,
               device:torch.device
               ):

    model.train()
    train_loss,train_acc=0,0
    for batch, (X_train,y_train) in enumerate(dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_class_logs= torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (pred_class_logs == y_train).sum().item() / len(y_train)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss,train_acc

def test_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               device:torch.device
               ):

    model.eval()
    test_loss,test_acc=0,0
    for batch, (X_test,y_test) in enumerate(dataloader):
        X_test, y_test = X_test.to(device), y_test.to(device)
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        test_loss += loss.item()
        pred_class_logs= torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        test_acc += (pred_class_logs == y_test).sum().item() / len(y_test)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss,test_acc

def train(model:torch.nn.Module,
        train_dataloader:torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          loss_fn:torch.nn.Module,
          epochs:int,
          device:torch.device,
          ):
    results ={
        'train loss':[],
        'test loss':[],
        'train acc':[],
        'test acc':[]
    }


    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           device=device
                                           )
        test_loss,test_acc = test_step(model=model,
                                       dataloader=test_dataloader,
                                       loss_fn=loss_fn,
                                       device=device)

        print(f' Epoch:{epoch+1} |')
        print(f'Train loss {train_loss} |')
        print(f'Test loss {test_loss:.2f} |')
        print(f'Train acc {train_acc} |')
        print(f'Test acc {test_acc:.2f} |')

        results['train loss'].append(train_loss)
        results['test loss'].append(test_loss)
        results['train acc'].append(train_acc)
        results['test acc'].append(test_acc)

    return results


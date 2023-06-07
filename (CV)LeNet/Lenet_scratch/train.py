import torch

def train_loop(train_dataloader, model, criterion, optimizer, device):
    model.train() # 모델을 학습 상태로 설정
    total_step = len(train_dataloader)
    for batch, (X, y) in enumerate(train_dataloader):
        # 데이터 디바이스에 담기
        X = X.to(device)
        y = y.to(device)

        # 예측값 계산 및 손실함수 계산(순전파)
        pred = model(X)
        loss = criterion(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0 :
            loss, current_step = loss.item(), batch
            print(f'[{current_step:>5d} / {total_step:5d}], loss : {loss:>7f}')

def test_loop(test_dataloader, model, device):
    print('********** start evaluation **********')
    model.eval() # 모델 평가 상태로 설정
    # testing
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


def training(num_epochs, train_dataloader, test_dataloader, model, criterion, optimizer, device):
    for epoch in range(1, num_epochs+1):
        print('='*30)
        print(f'epoch : [{epoch} / {num_epochs}]')
        print('='*30)
        train_loop(train_dataloader, model, criterion, optimizer, device)
        
        if epoch % 5 == 0 :
            test_loop(test_dataloader, model, device)

    print('done!')
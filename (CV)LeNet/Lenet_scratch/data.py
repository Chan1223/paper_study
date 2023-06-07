

def load_mnist(path = './MNIST'):
    # torchvision : 데이터셋 호출, 모델 아키텍쳐 가져오기, 이미지 변환 등을 지원받기 위해 사용
    from torchvision.datasets import MNIST
    from torchvision.transforms import transforms


    # MNIST의 데이터는 28*28 size지만, Lenet-5는 32*32 data를 위해 사용됨
    # resizing을 통해 input 값을 맞춰줌

    # transformation 정의하기
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
    ])

        # train / test dataset 호출하기

    train_data = MNIST(
        root=path,
        train=True,
        download=True,
        transform=data_transform
    )

    test_data = MNIST(
        root=path,
        train=False,
        download=True,
        transform=data_transform
    )

    return train_data, test_data

def make_dataloader(train_data, test_data, batch_size):
    # 데이터 로더 가져오기
    from torch.utils.data.dataloader import DataLoader

    # 다운받은 dataset을 DataLoader에 담기
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size = 1)

    data_info(train_dataloader, test_dataloader)

    return train_dataloader, test_dataloader

def data_info(train_dataloader, test_dataloader):
    print('train 데이터 개수 : ', len(train_dataloader))
    print('test 데이터 개수 : ', len(test_dataloader))
    print()
    print('=========data shape=========')

    for x,y in test_dataloader:    
        print('Shape of x {N,C,H,W}',x.shape)
        print('Shape of y:',y.shape,y.dtype)
        break
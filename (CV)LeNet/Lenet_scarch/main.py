# 실행하려면 아래의 코드 입력
# python main.py --model 'LeNet()'

import argparse
from torchsummary import summary

# 토치 불러오기
import torch
import torch.nn as nn

# torchvision : 데이터셋 호출, 모델 아키텍쳐 가져오기, 이미지 변환 등을 지원받기 위해 사용

# 최적화 알고리즘 가져오기
from model import LeNet, LeNet_with_sequential
from data import load_mnist, make_dataloader
from train import train_loop, test_loop, training

def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=['LeNet()', 'LeNet_with_sequential()'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('-size', '--batch_size', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=int, default=10)

    config = parser.parse_args()

    print(f'model : {config.model}')
    print(f'learning_rate : {config.learning_rate}')
    print(f'batch_size : {config.batch_size}')
    print(f'epochs : {config.epochs}')
    return config


def main(config):
    # 데이터 호출 
    train_data, test_data = load_mnist()
    train_dataloader, test_dataloader = make_dataloader(train_data, test_data, config.batch_size)

    # 모델 호출
    model = eval(config.model)

    # 학습
    learning_rate = config.learning_rate

    device = 'cuda'
    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters() ,lr=learning_rate)

    criterion.to(device)

    summary(model, input_size=(1, 32, 32))  

    training(config.epochs, train_dataloader, test_dataloader, model, criterion, optimizer, device)

if __name__=="__main__":
    config = define_argparser()
    main(config)
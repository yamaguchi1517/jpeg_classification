import argparse
import csv
from datetime import datetime
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from pathlib import Path
import random
from time import perf_counter
from tqdm import tqdm
import numpy as np

# PyTorchパッケージとそのサブモジュール
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

# 自作パッケージ
from models.classification_lstm import Classification_lstm
from utils.device import AutoDevice
from utils.transforms.to_tensor import ToTensor
from utils.transforms.encode import EncodeJPEG
from utils.transforms.add_noise import RandomMASK, MultNORM, UniformOR, RandomFLIP
from utils.transforms.padding import PadSequence

parser = argparse.ArgumentParser(
    prog='プライバシー保護に応用したJPEGデータの機械学習',
    description='PyTorchを用いてMNIST/FashionMNIST/cifar10の画像分類を行います。'
)
parser.add_argument(
    '--embedding_dim', help="分散表現のサイズを指定します(デフォルト: --embedding_dim 48)",
    type=int, default=48
)
parser.add_argument(
    '--hidden_dim', help="隠れ層の数を指定します(デフォルト: --hidden_dim 120)",
    type=int, default=120
)
parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=1000, metavar='B'
)
parser.add_argument(
    '-e', '--epochs', help='エポック数を指定します。',
    type=int, default=1000, metavar='E'
)
parser.add_argument(
    '--dataset', help='データセットを指定します。',
    type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'cifar10']
)
parser.add_argument(
    '--data-path', help='データセットのパスを指定します。',
    type=str, default='./data'
)
parser.add_argument(
    '--seed', help='乱数生成器のシード値を指定します。',
    type=int, default=42
)
parser.add_argument(
    '--save', help='訓練したモデルを保存します。',
    action='store_true'
)
parser.add_argument(
    '--save-interval', help='モデルの保存間隔をエポック数で指定します。',
    type=int, default=10,
)
parser.add_argument(
    '--load', help='訓練したモデルを読み込みます。',
    type=str, default=None
)
parser.add_argument(
    '--test', help='訓練を行わずテストのみ行います。',
    action='store_true'
)
parser.add_argument(
    '--disable-cuda', '--cpu', help='GPU上で計算せず、全てCPU上で計算します。',
    action='store_true'
)
parser.add_argument(
    '--info', help='ログ表示レベルをINFOに設定し、詳細なログを表示します。',
    action='store_true'
)
parser.add_argument(
    '--debug', help='ログ表示レベルをDEBUGに設定し、より詳細なログを表示します。',
    action='store_true'
)

args = parser.parse_args()

LAUNCH_DATETIME = datetime.now()

basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)

logger = getLogger('main')

# テスト時は何度エポックを回しても結果は同じなので、エポック数を1に強制する
if args.test:
    args.epochs = 1
    logger.info('テストモードで実行されているためエポック数を1に設定しました。')

OUTPUT_DIR = Path(
    LAUNCH_DATETIME.strftime(
        f'./outputs/{args.dataset}/%Y%m%d%H%M%S'))
OUTPUT_DIR.mkdir(parents=True)
logger.info(f'結果出力用のディレクトリ({OUTPUT_DIR})を作成しました。')
if args.save:
    OUTPUT_MODEL_DIR = OUTPUT_DIR.joinpath('models')
    OUTPUT_MODEL_DIR.mkdir(parents=True)
    logger.info(f'モデル保存用のディレクトリ({OUTPUT_MODEL_DIR})を作成しました。')

#乱数のシード値の設定
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# GPUで再現性を確保するために以下の2つを設定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
logger.info(f'ランダムシードを{args.seed}に設定しました。')

auto_device = AutoDevice(disable_cuda=args.disable_cuda)
device = auto_device()

train_tfs = []
test_tfs = []

train_tfs.extend([
    transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
    ToTensor(),
    EncodeJPEG(),
    #ビット変換を加える際は、以下のいずれか一つを用いる
    #RandomMASK()
    #MultNORM()
    #RandomFLIP()
])

test_tfs.extend([
    transforms.Resize(32),
    ToTensor(),
    EncodeJPEG(),
])

if args.dataset == 'mnist' or 'fashion_mnist':
    train_tfs.append(PadSequence(800))
    test_tfs.append(PadSequence(800))

if args.dataset == 'cifar10':
    train_tfs.insert(2, transforms.Normalize(
        [128, 128, 128],#RGB平均
        [0.5, 0.5, 0.5] #RGB標準偏差
    ))
    train_tfs.append(PadSequence(1280))

    test_tfs.insert(2, transforms.Normalize(
        [128, 128, 128],
        [0.5, 0.5, 0.5]
    ))
    test_tfs.append(PadSequence(1280))

logger.info('画像のトランスフォームを定義しました。')

def load_dataset(name: str, train_transform=None, test_transform=None):
    if isinstance(train_transform, (list, tuple)):
        train_transform = transforms.Compose(train_transform)
    if isinstance(test_transform, (list, tuple)):
        test_transform = transforms.Compose(test_transform)

    if name == 'mnist':
        num_classes = 10
        trainset = dset.MNIST(
            root=args.data_path, download=True, train=True,
            transform=train_transform)
        testset = dset.MNIST(
            root=args.data_path, download=True, train=False,
            transform=test_transform)
    elif name == 'fashion_mnist':
        num_classes = 10
        trainset = dset.FashionMNIST(
            root=args.data_path, download=True, train=True,
            transform=train_transform)
        testset = dset.FashionMNIST(
            root=args.data_path, download=True, train=False,
            transform=test_transform)
    elif name == 'cifar10':
        num_classes = 10
        trainset = dset.CIFAR10(
            root=args.data_path, download=True, train=True,
            transform=train_transform)
        testset = dset.CIFAR10(
            root=args.data_path, download=True, train=False,
            transform=test_transform)
        

    return trainset, testset, num_classes

trainset, testset, NUM_CLASSES = load_dataset(args.dataset, train_tfs, test_tfs)
logger.info('データセットを読み込みました。')

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False)
logger.info('データローダーを生成しました。')

# モデルの定義
model = Classification_lstm(
    embedding_dim = args.embedding_dim,
    hidden_dim = args.hidden_dim, 
    num_classes = NUM_CLASSES)
logger.info('モデルを定義しました。')

model = model.to(device)
logger.info(f'モデルを{device}に転送しました。')

if args.load is not None:
    model.load_state_dict(torch.load(args.load))
    logger.info(f'学習済みモデルを{args.load}から読み込みました。')


optimizer = optim.Adam(model.parameters(), lr=0.001)
logger.info('オプティマイザを定義しました。')

criterion = nn.CrossEntropyLoss()
logger.info('損失関数を定義しました。')

f_results = open(
    OUTPUT_DIR.joinpath('results.csv'), mode='w', encoding='utf-8')
logger.info('結果出力用のファイルを開きました。')

csv_writer = csv.writer(f_results, lineterminator='\n')

result_items = [
    'Epoch', 'Train Loss Mean', 'Train Accuracy Mean', 'Train Elapsed Time',
    'Test Loss Mean', 'Test Accuracy Mean', 'Test Elapsed Time',
    'Saved File Name',
]
# ファイルの一行目に項目名
csv_writer.writerow(result_items)
f_results.flush()

# それぞれの項目名が何番目の項目になったかを辞書に記録
csv_idx = {item: i for i, item in enumerate(result_items)}

logger.info('訓練を開始します。')

for epoch in range(args.epochs):
    results = ['' for _ in range(len(csv_idx))]
    results[csv_idx['Epoch']] = f'{epoch + 1}'

    #訓練モード
    if not args.test:
        model.train()
        pbar = tqdm(
            trainloader,
            desc=f'[{epoch + 1}/{args.epochs}] 訓練開始',
            total=len(trainset)//args.batch_size,
            leave=False)

        losses, cnt_total, cnt_correct = [], 0, 0
        begin_time = perf_counter()  # 時間計測開始
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            model.zero_grad()
            outputs = model(images)          
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            cnt_total += labels.size(0)
            cnt_correct += (predicted == labels).sum().item()

            losses.append(loss.item())
            pbar.set_description_str(
                f'[{epoch+1}/{args.epochs}] 訓練中... '
                f'<損失: {losses[-1]:.016f}>')
        end_time = perf_counter()
        pbar.close()

        #記録
        train_loss_mean = np.mean(losses)
        results[csv_idx['Train Loss Mean']] = f'{train_loss_mean:.016f}'

        train_accuracy_mean = 100 * cnt_correct / cnt_total
        results[csv_idx['Train Accuracy Mean']] = f'{train_accuracy_mean:.03f}'

        train_elapsed_time = end_time - begin_time
        results[csv_idx['Train Elapsed Time']] = f'{train_elapsed_time:.07f}'

    model.eval()
    with torch.no_grad():
        pbar = tqdm(
            enumerate(testloader),
            desc=f'[{epoch + 1}/{args.epochs}] テスト開始',
            total=len(testset)//args.batch_size,
            leave=False)
        losses, cnt_total, cnt_correct = [], 0, 0
        begin_time = perf_counter()
        for i, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            cnt_total += labels.size(0)
            cnt_correct += (predicted == labels).sum().item()
            pbar.set_description_str(
                f'[{epoch+1}/{args.epochs}] テスト中... '
                f'<損失: {losses[-1]:.016f}>')
        end_time = perf_counter()
    test_loss_mean = np.mean(losses)
    results[csv_idx['Test Loss Mean']] = f'{test_loss_mean:.016f}'

    test_accuracy_mean = 100 * cnt_correct / cnt_total
    results[csv_idx['Test Accuracy Mean']] = f'{test_accuracy_mean:.03f}'

    test_elapsed_time = end_time - begin_time
    results[csv_idx['Test Elapsed Time']] = f'{test_elapsed_time:.07f}'

    if args.save and (
            (epoch + 1) % args.save_interval == 0
            or epoch + 1 == args.epochs):
        print(f'[{epoch+1}/{args.epochs}] モデルの保存中... ', end='')
        saved_file_name = OUTPUT_MODEL_DIR.joinpath(f'model_{epoch+1:06d}.pt')
        torch.save(model.state_dict(), saved_file_name)
        results[csv_idx['Saved File Name']] = saved_file_name
        print('<完了>')

    csv_writer.writerow(results)
    f_results.flush()

    if not args.test:
        print(
            f'[{epoch+1}/{args.epochs}] 訓練完了. '
            f'<訓練: (経過時間: {train_elapsed_time:.03f}[s/epoch]'
            f', 平均損失: {train_loss_mean:.05f}'
            f', 平均正解率: {train_accuracy_mean:.02f}[%])'
            f', テスト: (経過時間: {test_elapsed_time:.03f}[s/epoch]'
            f', 平均損失: {test_loss_mean:.05f}'
            f', 平均正解率: {test_accuracy_mean:.02f}[%])>')
    else:
        print(
            f'[{epoch+1}/{args.epochs}] テスト完了. '
            f'<テスト: (経過時間: {test_elapsed_time:.03f}[s/epoch]'
            f', 平均損失: {test_loss_mean:.05f}'
            f', 平均正解率: {test_accuracy_mean:.02f}[%])>')

logger.info('実行が終了しました。')
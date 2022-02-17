## 2주차 정리

<br />

### 목차
[1. PyTorch](#PyTorch)  
[2. PyTorch Bascis](#PyTorch-Basics)  
[3. AutoGrad & Optimizer](#AutoGrad-&-Optimizer)  
[4. Datasets & DataLoaders](#Datasets-&-DataLoaders)  
[5. 모델 불러오기](#모델-불러오기)  
[6. Monitoring tools for PyTorch](#Monitoring-tools-for-PyTorch)  
[7. Multi-GPU 학습](Multi-GPU-학습)  
[8. Hyperparameter Tuning](#Hyperparameter-Tuning)  

<br />

### PyTorch
* PyTorch와 TensorFlow의 차이점:
  * PyTorch 
    * ```Define by Run(Dynamic Computational Graph, DCG)```, 실행을 하면서 그래프를 생성하는 방식
    * 즉시 확인 가능 -> pythonic code
  * TensorFlow 
    * ```Define and Run(Static Graphs)```, 그래프를 먼저 정의 후 실행시점에 데이터 feed
    * 그래프를 한 번 빌드하고, 반복해서 실행함
* PyTorch의 장점:
  * Numpy 구조를 가지는 Tensor 객체로 array를 표현함
  * 자동미분을 자원하여 DL 연산을 지원함
  * 다양한 형태의 DL을 지원하는 함수와 모델을 지원함

<br />

### PyTorch Basics
* ```numpy```의 ```ndarray```와 개념상 동일한 ```PyTorch``` 클래스 ```tensor```를 사용함
``` python
import numpy as np
import torch

n_array = np.arange(10).reshape(2, 5)
t_array = torch.FloatTensor(n_array)
```

* ```Pytorch```의 ```tensor```는 GPU에 올려서 사용 가능
``` python
x_data.device
# device(type='cpu)

if torch.cuda.is_available():
  x_data_cuda = x_data.to('cuda')
x_data_cuda.device
# device(type='cuda', index=0)
```

* ```view```를 통하여 tensor의 shape을 변환
  * ```PyTorch```에선 ```reshape``` 대신에 ```view``` 사용을 권장
  * ```view```와 ```reshape```은 contiguity 보장의 차이

* ```squeeze```: 차원의 개수가 1인 차원을 삭제(압축)
* ```unsqueeze```: 차원의 개수가 1인 차원을 추가
``` python
import torch

tensor_ex = torch.rand(size=(2, 1, 2))
tensor_ex.squeeze()
# tensor([[0.2245, 0.9825],
#         [0.0646, 0.7184]])

tensor_ex = torch.rand(size=(2, 2)) 
tensor_ex.unsqueeze(0).shape
# torch.Size([1, 2, 2]) 
tensor_ex.unsqueeze(1).shape
  # torch.Size([2, 1, 2])
tensor_ex.unsqueeze(2).shape 
# torch.Size([2, 2, 1])
```

* 행렬곱셈 연산의 함수는 ```dot``` 이 아닌 ```mm```사용
* ```mm``` 과 ```matmul``` 은 broadcasting을 지원한다. 
``` python
import torch

n1 = torch.rand(size=(2, 3))
n2 = torch.rand(size=(3, 5))

n1.mm(n2)
n1.matmul(n2)
```

<br />

### AutoGrad & Optimizer
* AutoGrad란 PyTorch의 **Auto**matic differentiation(**Grad**ient) package이다. 즉, backpropagation을 자동으로 해준다.
* ```torch.nn.Module```: 딥러닝을 구성하는 Layer의 base class이다. ```Input```, ```Output```, ```Forward```, ```Backward```를 정의하며, 학습의 대상이 되는 ```parameter(tensor)```를 정의한다.
* ```torch.nn.Parameter```: ```Tensor``` 객체의 상속 객체이다. ```nn.Module```내에 attribute가 될 때는 ```required_grad = True```로 지정되어 학습 대상이 되는 ```Tensor```이다.
* ```Backward```: Layer에 있는 ```Parameter```들의 미분을 수행한다. ```Forward```의 결과값과 실제값간의 차이(loss)에 대하여 미분을 수행한다. 해당 값으로 ```Parameter```를 업데이트한다.

<br />

### Datasets & DataLoaders
* Dataset 클래스:
  * 데이터 입력 형태를 정의하는 클래스
  * 데이터를 입력하는 방식의 표준화

``` python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self,):
    '''
    데이터의 위치나 파일명과 같은 초기화 작업을 위해 동작
    '''
    pass

  def __len__(self):
    '''
    Dataset의 최대 요소 수를 반환하는데 사용
    '''
    pass

  def __getitem__(self, idx):
    '''
    데이터셋의 idx번째 데이터를 반환하는데 사용
    '''
    pass

dataset_custom = CustomDataset()
```

* DataLoader 클래스:
  * Data의 Batch를 생성해주는 클래스
  * 학습직전 데이터의 변환을 책임
  * Tensor로 변환 + Batch 처리가 메인 업무

``` python
DataLoader(dataset,            # Dataset 인스턴스가 들어감
           batch_size=1,       # 배치 사이즈를 설정
           shuffle=False,      # 데이터를 섞어서 사용하겠는지를 설정
           sampler=None,       # sampler는 index를 컨트롤
           batch_sampler=None, # 위와 비슷하므로 생략
           num_workers=0,      # 데이터를 불러올때 사용하는 서브 프로세스 개수
           collate_fn=None,    # map-style 데이터셋에서 sample list를 batch 단위로 바꾸기 위해 필요한 기능
           pin_memory=False,   # Tensor를 CUDA 고정 메모리에 할당
           drop_last=False,    # 마지막 batch를 사용 여부
           timeout=0,          # data를 불러오는데 제한시간
           worker_init_fn=None # 어떤 worker를 불러올 것인가를 리스트로 전달
          )

dataloader_custom = DataLoader(dataset_custom)
```

* 일반적인 학습 과정
  * 아래와 같은 구조를 ```train.py``` 에 넣어서 한꺼번에 실행시킨다.
``` python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from network import CustomNet
from dataset import ExampleDataset
from loss import ExampleLoss

###############################
# Custom modeling #
###############################

# 모델 생성
model = CustomNet()
model.train()

# 옵티마이저 정의
params = [param for param in model.parameters() if param.requires_grad]
optimizer = optim.Example(params, lr=lr)

# 손실함수 정의
loss_fn = ExampleLoss()

###########################################
# Custom Dataset & DataLoader #
###########################################

# 학습을 위한 데이터셋 생성
dataset_example = ExampleDataset()

# 학습을 위한 데이터로더 생성
dataloader_example = DataLoader(dataset_example)

##########################################################
# Transfer Learning & Hyper Parameter Tuning # 
##########################################################
for e in range(epochs):
  for X,y in dataloader_example:
    output = model(X)
    loss = loss_fn(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

<br />

### 모델 불러오기
##### 학습 결과를 공유하기 위하여 학습 결과를 저장할 필요가 있다.
* ```model.save()```:
  * 학습의 결과를 저장하기 위한 함수
  * 모델 형태(architecture)와 파라미터를 저장
  * 모델 학습 중간 과정의 저장을 통하여 최선의 결과 모델을 선택
  * 만들어진 모델을 외부 연구자와 공유하여 학습 재연성 향상
``` python
print("Model's state_dict: ")
for param_tensor in model.state_dict():       # state_dict: 모델의 파라미터를 표시
  print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# 모델의 파라미터를 저장
torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model.pt"))

# 같은 모델의 형태에서 파라미터만 load
new_model = TheModelClass()
new_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pt")))

# 모델의 architecture와 함께 저장
torch.save(model, os.path.join(MODEL_PATH, "model.pt"))

# 모델의 architecture와 함께 load
model = torch.load(os.path.join(MODEL_PATH, "model.pt"))
```

<br />

* checkpoints
  * 학습의 중간 결과를 저장하여 최선의 결과를 선택
  * earlystopping 기법 사용시 이전 학습의 결과물을 저장
  * loss와 metric값을 지속적으로 확인 저장
  * 일반적으로 epoch, loss, metric을 함께 저장하여 확인
  * colab에서 지속적인 학습을 위해 필요
``` python
torch.save({ 
  'epoch': e,       # 모델의 정보를 epoch과 함께 저장
  'model_state_dict': model.state_dict(), 
  'optimizer_state_dict': optimizer.state_dict(), 
  'loss': epoch_loss,
}, f"saved/checkpoint_model_{e}_{epoch_loss/len(dataloader)}_{epoch_acc/len(dataloader)}.pt")

checkpoint = torch.load(PATH) 
model.load_state_dict(checkpoint['model_state_dict']) 
optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

<br />

* pretrained model Transfer learning
  * 남이 만든 모델을 쓰고싶을 때
  * 다른 데이터셋으로 만든 모델을 현재 데이터에 적용
  * 일반적으로 대용량 데이터셋으로 만들어진 모델의 성능이 높다
  * 현재의 DL에서는 가장 일반적인 학습 기법
  * backbone architecture가 잘 학습된 모델에서 일부분만 변경하여 학습을 수행함
``` python
# vgg16 모델을 vgg에 할당하기
vgg = models.vgg16(pretrained=True).to(device)

class MyNewNet(nn.Module): 
  def __init__(self):
    super(MyNewNet, self).__init__() 
    self.vgg19 = models.vgg19(pretrained=True) 
    self.linear_layers = nn.Linear(1000, 1)     # 모델에 마지막 Linear Layer 추가

  # Defining the forward pass
  def forward(self, x):
    x = self.vgg19(x)
    return self.linear_layers(x)

for param in my_model.parameters(): 
  param.requires_grad = False     # 마지막 레이어를 제외하고 frozen
for param in my_model.linear_layers.parameters(): 
  param.requires_grad = True
```

<br />

### Monitoring tools for PyTorch
  * 학습 데이터를 모니터링할 수 있는 도구들
##### Tensorboard
  * TensorFlow의 프로젝트로 만들어진 시각화 도구
  * 학습 그래프, metric, 학습 결과의 시각화 지원
  * PyTorch도 연결 가능 -> DL 시각화 핵심 도구
``` python
import os
logs_base_dir = "logs"
os.makedirs(logs_base_dir, exist_ok=True)       # Tensorboard 기록을 위한 directory 생성

from torch.utils.tensorboard import SummaryWriter       # 기록 생성 객체 SummaryWriter 생성
import numpy as np

writer = SummaryWriter(logs_base_dir)
for n_iter in range(100):
  writer.add_scalar('Loss/train', np.random.random(), n_iter)           # add_scalar 함수 : scalar 값을 기록
  writer.add_scalar('Loss/test', np.random.random(), n_iter)            # Loss/train : loss category에 train 값
  writer.add_scalar('Accuracy/train', np.random.random(), n_iter)       # n_iter : x 축의 값
  writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
writer.flush()        # 값 기록 (disk에 쓰기)

# jupyter 상에서 tensorboard 수행 
# 파일 위치 지정 (logs_base_dir) 같은 명령어를 콘솔에서도 사용가능
%load_ext tensorboard
%tensorboard --logdir {logs_base_dir}
```
##### weight & biases
  * 머신러닝 실험을 원활히 지원하기 위한 상용도구
  * 협업, code versioning, 실험 결과 기록 등 제공
  * MLOps의 대표적인 툴로 저변 확대 중
``` python
!pip install wandb -q

config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate" : LEARNING_RATE} 
wandb.init(project="my-test-project", config=config)
# wandb.config.batch_size = BATCH_SIZE            config 설정
# wandb.config.learning_rate = LEARNING_RATE

for e in range(1, EPOCHS+1):
  epoch_loss = 0
  epoch_acc = 0
  for X_batch, y_batch in train_dataset:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device).type(torch.cuda.FloatTensor)
    # ...
    optimizer.step()

    # ...

  wandb.log({'accuracy': train_acc, 'loss': train_loss})        # 기록 add_~~~ 함수와 동일
```

<br />

### Multi-GPU 학습
오늘날의 딥러닝은 엄청난 양의 데이터 처리를 필요로한다.
* Model parallel
  * 다중 GPU에 모델을 나눠 학습을 분산하는 방법
  * 모델을 나누는 것은 생각보다 예전부터 썼다(alexnet)
  * 모델의 병복, 파이프라인의 어려움 등으로 인하여 모델 병렬화는 고난이도 과제
``` python
class ModelParallelResNet50(ResNet):
  def __init__(self, *args, **kwargs):
    super(ModelParallelResNet50, self).__init__(
      Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

    self.seq1 = nn.Sequential(
      self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2
    ).to('cuda:0')        # 첫번째 모델을 cuda 0에 할당

    self.seq2 = nn.Sequential(
      self.layer3, self.layer4, self.avgpool,
    ).to('cuda:1')        # 두번째 모델을 cuda 0에 할당

    self.fc.to('cuda:1')

def forward(self, x):
  x = self.seq2(self.seq1(x).to('cuda:1'))        # 두 모델을 연결하기
  return self.fc(x.view(x.size(0), -1))
```

* Data parallel
  * 데이터를 나누어 GPU에 할당후 결과의 평균을 취하는 방법
  * minibatch 수식과 유사하지만 한번에 여러 GPU에서 수행한다
  * PyTorch에서는 아래 두 가지 방식을 제공
    * ```DataParallel```
      * 단순히 데이터를 분배한 후 평균을 취함
      * GPU 사용 불균형 문제 발생, Batch 사이즈 감소 (한 GPU가 병목), GIL
    ``` python
    parallel_model = torch.nn.DataParallel(model)       # Escapsulate the model

    predictions = parallel_model(inputs)        # Forward
    # pass on multi-GPUs
    loss = loss_function(predictions, labels)       # Compute
    #loss function
    loss.mean().backward()        # Average
    #GPU-losses + backward pass
    optimizer.step()
    predictions = parallel_model(inputs)        # Forward
    # pass with new parameters
    ```
    * ```DistributedDataParallel```
      * 각 CPU마다 process 생성하여 개별 GPU에 할당
      * 기본적으로 DataParallel로 하나 개별적으로 연산의 평균을 냄
    ``` python
    # Sampler를 사용
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    shuffle = False
    pin_memory = True

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, 
                  shuffle=True pin_memory=pin_memory, num_workers=3, shuffle=shuffle, 
                  sampler=train_sampler)

    def main():
      n_gpus = torch.cuda.device_count() torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, ))

    def main_worker(gpu, n_gpus):
      image_size = 224
      batch_size = 512
      num_worker = 8
      epochs = ...

      batch_size = int(batch_size / n_gpus)
      num_worker = int(num_worker / n_gpus)

      # 멀티프로세싱 통신 규약 정의
      torch.distributed.init_process_group(backend='nccl’, init_method='tcp://127.0.0.1:2568’, world_size=n_gpus, rank=gpu)

      model = MODEL

      torch.cuda.set_device(gpu)
      model = model.cuda(gpu)
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])        # Distributed datataparallel 정의
    ```
      * 파이썬의 멀티프로세싱 코드
      ``` python
      from multiprocessing import Pool

      def f(x):
        return x*x

      if __name__ == '__main__': 
        with Pool(5) as p:
          print(p.map(f, [1, 2, 3]))
      ```

<br />

### Hyperparameter Tuning
모델 스스로 학습하지 않는 값을 사람이 지정(learning rate, 모델의 크기, optimizer 등)
* Ray
  * multi-node multi processing 지원 모듈
  * ML/DL의 병렬 처리를 위해 개발된 모듈
  * 기본적으로 현재의 분산병렬 ML/DL 모듈의 표준
  * Hyperparameter Search를 위한 다양한 모듈 제공
``` python
data_dir = os.path.abspath("./data") 
load_data(data_dir)
# config에 search space 지정
config = {
  "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)), 
  "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
  "lr": tune.loguniform(1e-4, 1e-1),
  "batch_size": tune.choice([2, 4, 8, 16]) 
}
# 학습 스케줄링 알고리즘 지정
scheduler = ASHAScheduler(
  metric="loss", mode="min", max_t=max_num_epochs, grace_period=1,
reduction_factor=2) 
# 결과 출력 양식 지정
reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

# 병렬 처리 양식으로 학습 진행
result = tune.run(
  partial(train_cifar, data_dir=data_dir), 
  resources_per_trial={"cpu": 2, "gpu": gpus_per_trial}, 
  config=config, num_samples=num_samples, 
  scheduler=scheduler,
  progress_reporter=reporter)
```

<br />
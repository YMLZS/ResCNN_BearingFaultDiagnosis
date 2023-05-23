import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import resnet18
from data_set import MyDataset


def train(model_save_path, train_result_path, val_result_path, hp_save_path, snr, epochs=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 定义参数
    batch_size = 128
    lr = 1E-5

    # 保存超参数
    hyper_parameters = {'学习率: ': '{}'.format(lr),
                        'batch_size: ': '{}'.format(batch_size)}
    fs = open(hp_save_path, 'w')
    fs.write(str(hyper_parameters))
    fs.close()

    # 加载数据
    train_path = r'F:\PyCharmWorkSpace\MultiFD\data\cu_data_noisy\{}db\train\train.csv'.format(snr)
    val_path = r'F:\PyCharmWorkSpace\MultiFD\data\cu_data_noisy\{}db\val\val.csv'.format(snr)
    train_dataset = MyDataset(train_path, 'fd')
    val_dataset = MyDataset(val_path, 'fd')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 定义模型
    model = resnet18()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 10)
    model.to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    best_acc_fd = 0.0
    train_result = []
    result_train_loss = []
    result_train_acc = []
    val_result = []
    result_val_loss = []
    result_val_acc = []
    #训练
    for epoch in range(epochs):
        #train
        train_loss = []
        train_acc = []
        model.train()
        train_bar = tqdm(train_loader)
        for datas, labels in train_bar:
            optimizer.zero_grad()
            outputs = model(datas.float().to(device))
            loss = criterion(outputs, labels.type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()

            # torch.argmax(dim=-1), 求每一行最大的列序号
            acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy
            train_loss.append(loss.item())
            train_acc.append(acc)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss.item())

        #val
        model.eval()
        valid_loss = []
        valid_acc = []
        val_bar = tqdm(val_loader)
        for datas, labels in val_bar:
            with torch.no_grad():
                outputs = model(datas.float().to(device))
            loss = criterion(outputs, labels.type(torch.LongTensor).to(device))

            acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy
            valid_loss.append(loss.item())
            valid_acc.append(acc)
            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        print(f"[{epoch + 1:02d}/{epochs:02d}] train loss = "
              f"{sum(train_loss) / len(train_loss):.5f}, train acc = {sum(train_acc) / len(train_acc):.5f}", end="  ")
        print(f"valid loss = {sum(valid_loss) / len(valid_loss):.5f}, valid acc = {sum(valid_acc) / len(valid_acc):.5f}")

        result_train_loss.append(sum(train_loss) / len(train_loss))
        result_train_acc.append((sum(train_acc) / len(train_acc)).item())
        result_val_loss.append(sum(valid_loss) / len(valid_loss))
        result_val_acc.append((sum(valid_acc) / len(valid_acc)).item())

        if best_acc_fd <= sum(valid_acc) / len(valid_acc):
            best_acc_fd = sum(valid_acc) / len(valid_acc)
            torch.save(model.state_dict(), model_save_path)

    train_result.append(result_train_loss)
    train_result.append(result_train_acc)
    val_result.append(result_val_loss)
    val_result.append(result_val_acc)

    np.savetxt(train_result_path, np.array(train_result), fmt='%.5f', delimiter=',')
    np.savetxt(val_result_path, np.array(val_result), fmt='%.5f', delimiter=',')


if __name__ == '__main__':
    group_index = 2
    for i in range(5):
        # cwru_data
        model_save_path = "result/result_cu_noisy/group{}/exp0{}/model.pth".format(group_index, i+1)
        hp_save_path = "result/result_cu_noisy/group{}/parameters.txt".format(group_index)
        train_result_path = "result/result_cu_noisy/group{}/exp0{}/train_result.txt".format(group_index, i+1)
        val_result_path = "result/result_cu_noisy/group{}/exp0{}/val_result.txt".format(group_index, i+1)
        train(model_save_path, train_result_path, val_result_path, hp_save_path, snr=-1)

    # snr = [1, -1, 5, -5]
    # group_index = [1, 2, 3]
    # for k in range(4):
    #     for i in range(5):
    #         # cwru_data
    #         model_save_path = "result/result_cu_noisy/group{}/exp0{}/model.pth".format(group_index[k], i+1)
    #         hp_save_path = "result/result_cu_noisy/group{}/parameters.txt".format(group_index[k])
    #         train_result_path = "result/result_cu_noisy/group{}/exp0{}/train_result.txt".format(group_index[k], i+1)
    #         val_result_path = "result/result_cu_noisy/group{}/exp0{}/val_result.txt".format(group_index[k], i+1)
    #         train(model_save_path, train_result_path, val_result_path, hp_save_path, snr=snr[k])

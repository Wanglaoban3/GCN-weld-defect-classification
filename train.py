import os
import argparse
import time
from openpyxl import Workbook
from utils.run import run



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="C:/wrd/ss304/ss304")
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--model_path', type=str, default='logs_ss304/yuxunlian.pth')
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--model_type', type=str, default='resnet18')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--filename', type=str, default='')
    opt = parser.parse_args()
    fine_tune_layer = ['linear.weight', 'linear.bias']
    my_run = run(opt.root_path, opt.num_classes, opt.batch_size,
                 opt.model_type, opt.model_path, opt.pretrained, 'Adam')

    timeStamp = time.time()  # time.time()返回的是一个时间戳，以1970年1月1日到目前为止的秒数
    timeArray = time.localtime(timeStamp)  # localtime()是转化为年月日分秒的格式
    otherStyleTime = time.strftime("%Y_%m_%d %H_%M_%S", timeArray)  # strftime转化为字符串格式
    print('开始训练', opt.filename)

    max_test_acc = 0
    if opt.pretrained:
        my_run.fine_tune(fine_tune_layer)
    for epoch in range(0, opt.epoch):  #开始训练
        start_time = time.time()
        my_run.train_one_epoch()
        my_run.eval_model()

        if my_run.epoch_results['test_acc'][epoch] > max_test_acc:
            max_test_acc = my_run.epoch_results['test_acc'][epoch]
            # best_weight = deepcopy(my_run.model)
            # best_weight = best_weight.cpu().state_dict()
        end_time = time.time()
        consume_time = round(end_time - start_time, 1)

        print(f'[epoch: {epoch+1}/{opt.epoch}, eta: {round(consume_time*(opt.epoch-epoch), 1)},'
              f'train_acc: {round(my_run.epoch_results["train_acc"][epoch], 3)}, train_loss: {round(my_run.epoch_results["train_loss"][epoch], 4)}, '
              f'test_acc: {round(my_run.epoch_results["test_acc"][epoch], 3)}, test_loss: {round(my_run.epoch_results["test_loss"][epoch], 4)}, '
              f'max: {round(max_test_acc, 3)}, lr: {round(my_run.scheduler.get_last_lr()[0], 5)}]')
        my_run.epoch += 1

    file_name = [opt.filename, opt.model_type, str(max_test_acc), otherStyleTime]
    file_name = '_'.join(file_name)
    save_path = './logs_al/' + file_name
    os.makedirs(save_path, exist_ok=True)
    # torch.save(best_weight, save_path+'/best_weight.pth')
    wb = Workbook()
    ws = wb.active
    ws.append(['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss'])
    for i in range(opt.epoch):
        ws.append([i+1, my_run.epoch_results["train_acc"][i], my_run.epoch_results["train_loss"][i],
                   my_run.epoch_results["test_acc"][i], my_run.epoch_results["test_loss"][i]])
    wb.save(save_path+'/datas.xlsx')
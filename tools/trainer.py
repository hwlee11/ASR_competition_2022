import torch
import numpy as np
import math
from dataclasses import dataclass
import time
from nsml import DATASET_PATH


def trainer(mode, config, dataloader, optimizer, model, criterion, metric, train_begin_time, device):


    if mode == 'train':
        log_format = "[INFO] step: {:4d}/{:4d}, loss: {:.6f}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    else:
        log_format = "[INFO] step: {:4d}/{:4d}, " \
                              "cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h, lr: {:.6f}"
    total_num = 0
    epoch_loss_total = 0.
    print(f'[INFO] {mode} Start')
    epoch_begin_time = time.time()
    cnt = 0
    cer = 1
    for inputs, targets, input_lengths, target_lengths in dataloader:
        begin_time = time.time()

        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = torch.as_tensor(target_lengths).to(device)
        model = model.to(device)


        if mode == 'train':
            if config.decoder is None:
                outputs, output_lengths = model(inputs, input_lengths,targets,target_lengths)
                loss = criterion(
                    outputs.transpose(0, 1),
                    targets[:, 1:],
                    tuple(output_lengths),
                    tuple(target_lengths)
                )

            elif config.decoder == 'rnnt':
                    outputs,output_lengths = model(inputs, input_lengths,targets,target_lengths)
                    loss = criterion(
                    outputs,
                    targets[:, 1:].contiguous().int(),
                    #targets.int(),
                    #targets[:, 1:].transpose(0, 1).contiguous().int(),
                    output_lengths.int(),
                    target_lengths.int()
                )
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step(model)
        else:
            if isinstance(model, nn.DataParallel):
                y_hats = model.module.recognize(inputs, input_lengths)
            else:
                y_hats = model.recognize(inputs, input_lengths)
            #y_hats = model.recognize(input

        total_num += int(input_lengths.sum())
        epoch_loss_total += loss.item()

        torch.cuda.empty_cache()

        if cnt % config.print_every:

            current_time = time.time()
            elapsed = current_time - begin_time
            epoch_elapsed = (current_time - epoch_begin_time) / 60.0
            train_elapsed = (current_time - train_begin_time) / 3600.0
            if mode == 'train':
                #cer = metric(targets[:, 1:], y_hats)
                print(log_format.format(
                    cnt, len(dataloader), loss,
                    cer, elapsed, epoch_elapsed, train_elapsed,
                    optimizer.get_lr(),
                ))
            else:
                cer = metric(targets[:, 1:], y_hats)
                print(log_format.format(
                    cnt, len(dataloader),
                    cer, elapsed, epoch_elapsed, train_elapsed,
                    optimizer.get_lr(),
                ))
        cnt += 1
    #return model, epoch_loss_total/len(dataloader), metric(targets[:, 1:], y_hats)

    if mode == 'train':
        return model, epoch_loss_total/len(dataloader), cer
    else:
        return model, cer

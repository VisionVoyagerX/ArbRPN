from pathlib import Path

import torch
from torch.optim import Adam
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

from data_loader.DataLoader import GaoFen2, WV3, Sev2Mod
from models.ArbRPN import ArbRPN
from utils import *
import matplotlib.pyplot as plt
import numpy as np


def main():

    choose_dataset = 'GaoFen2'  # choose dataset

    if choose_dataset == 'GaoFen2':
        dataset = eval('GaoFen2')
        tr_dir = '../pansharpenning_dataset/GF2/train/train_gf2.h5'
        eval_dir = '../pansharpenning_dataset/GF2/val/valid_gf2.h5'
        test_dir = '../pansharpenning_dataset/GF2/test/test_gf2_multiExm1.h5'
        checkpoint_dir = 'checkpoints/ArbRPN_GF2/ArbRPN_GF2_2023_07_27-15_22_08.pth.tar'
        ms_channel = 4
        ergas_l = 4
    elif choose_dataset == 'WV3':
        dataset = eval('WV3')
        tr_dir = '../pansharpenning_dataset/WV3/train/train_wv3.h5'
        eval_dir = '../pansharpenning_dataset/WV3/val/valid_wv3.h5'
        test_dir = '../pansharpenning_dataset/WV3/test/test_wv3_multiExm1.h5'
        checkpoint_dir = 'checkpoints/ArbRPN_WV3/ArbRPN_WV3_2023_07_28-01_12_12.pth.tar'
        ms_channel = 8
        ergas_l = 4
    else:
        print(choose_dataset, ' does not exist')

    # Prepare device
    # TODO add more code for server
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize DataLoader
    train_dataset = dataset(
        Path(tr_dir), transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])  # /home/ubuntu/project
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, drop_last=True)

    validation_dataset = dataset(
        Path(eval_dir))
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=64, shuffle=True)

    test_dataset = dataset(
        Path(test_dir))
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics
    model = ArbRPN(mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                   pan_std=train_dataset.pan_std.to(device)).to(device)

    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0)

    criterion = L1Loss().to(device)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    tr_report_loss = 0
    val_report_loss = 0
    test_report_loss = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []

    ergas_score = 0
    sam_score = 0
    q2n_score = 0
    psnr_score = 0
    ssim_score = 0
    
    best_eval_psnr = 0
    best_test_psnr = 0
    current_daytime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    steps = 200000
    save_interval = 1000
    report_interval = 50
    test_intervals = [25000, 50000, 75000, 100000, 125000,
                      150000, 175000, 200000]
    evaluation_interval = [25000, 50000, 75000, 100000, 125000,
                           150000, 175000, 200000]

    val_steps = 100

    # Model summary
    summary(model, [(1, 1, 256, 256), (1, ms_channel, 64, 64)],
            dtypes=[torch.float32, torch.float32])

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    lr_decay_intervals = [50000, 100000, 15000, 175000]
    continue_from_checkpoint = True

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics = load_checkpoint(torch.load(
            checkpoint_dir), model, optimizer, tr_metrics, val_metrics)
        print('Model Loaded ...')

    def scaleMinMax(x):
        return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))

    # evaluation mode
    model.eval()
    with torch.no_grad():
        test_iterator = iter(test_loader)
        for i, (pan, mslr, mshr) in enumerate(test_iterator):
            # forward
            pan, mslr, mshr = pan.to(device), mslr.to(
                device), mshr.to(device)
            mssr = model(pan, mslr)
            test_loss = criterion(mssr, mshr)
            test_metric = test_metric_collection.forward(mssr, mshr)
            test_report_loss += test_loss

            # compute metrics
            test_metric = test_metric_collection.compute()
            test_metric_collection.reset()

            
            ergas_score += ergas_batch(mshr, mssr, ergas_l)
            sam_score += sam_batch(mshr, mssr)
            q2n_score += q2n_batch(mshr, mssr)
            psnr_score += test_metric['psnr'].item()
            ssim_score += test_metric['ssim'].item()

            figure, axis = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
            axis[0].imshow((scaleMinMax(mslr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[0].set_title('(a) LR')
            axis[0].axis("off")

            axis[1].imshow(pan.permute(0, 3, 2, 1).detach().cpu()[
                0, ...], cmap='gray')
            axis[1].set_title('(b) PAN')
            axis[1].axis("off")

            axis[2].imshow((scaleMinMax(mssr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[2].set_title(
                f'(c) ArbRPN {test_metric["psnr"]:.2f}dB/{test_metric["ssim"]:.4f}')
            axis[2].axis("off")

            axis[3].imshow((scaleMinMax(mshr.permute(0, 3, 2, 1).detach().cpu()[
                            0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
            axis[3].set_title('(d) GT')
            axis[3].axis("off")

            plt.savefig(f'results/Images_{choose_dataset}_{i}.png')

            mslr = mslr.permute(0, 3, 2, 1).detach().cpu().numpy()
            pan = pan.permute(0, 3, 2, 1).detach().cpu().numpy()
            mssr = mssr.permute(0, 3, 2, 1).detach().cpu().numpy()
            gt = mshr.permute(0, 3, 2, 1).detach().cpu().numpy()

            np.savez(f'results/img_array_{choose_dataset}_{i}.npz', mslr=mslr,
                     pan=pan, mssr=mssr, gt=gt)

        # Print final scores
        print(f"Final scores:\n"
                f"ERGAS: {ergas_score / (i+1)}\n"
                f"SAM: {sam_score / (i+1)}\n"
                f"Q2n: {q2n_score / (i+1)}\n"
                f"PSNR: {psnr_score / (i+1)}\n"
                f"SSIM: {ssim_score / (i+1)}")

if __name__ == '__main__':
    main()

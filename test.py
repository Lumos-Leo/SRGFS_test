import torch
import os
import numpy as np
import math
import logging

from PIL import Image
from time import time
from models import *
from imageUtils.ImageUtils import *
from options_ import args
from Utils.CommonUtils import AverageMeter, computer_psnr, calc_ssim, remove_prefix, imresize_np
from Data import Data
from Logger import Logger
from torchvision import transforms

def test_snap_sp(net, valid_loader, args, device, dataset='', savefig=False):
        net.eval()
        psnr_mean = AverageMeter('PSNR', ':2f')
        ssim_mean = AverageMeter('SSIM', ':4f')
        with torch.no_grad():
            for i,data in enumerate(valid_loader):
                valid_y, label_y = data
                valid_y = valid_y.to(device)
                label_y = label_y.to(device)

                # compute output
                meta = {'masks': [], 'general_mask':[], 'features':[], 'layer_features':[], 'tail_masks':[]}
                pre_y,meta = net((valid_y, meta))
                pre_y = pre_y.clamp(0, 1)

                psnr = computer_psnr(pre_y.clamp(0,1), label_y, args.scale, data_range=1.)
                ssim = calc_ssim(pre_y, label_y, 1.)

                psnr_mean.update(psnr.item(), len(valid_y))
                ssim_mean.update(ssim.item(), len(valid_y))

                # if self.rank == 0:
                #     progress.display(i)
                if savefig:
                    import os
                    path = os.path.join('./test_results', dataset)
                    if not os.path.exists(path):
                        os.mkdir(path)
                    transforms.ToPILImage()(pre_y[0,:,:,:]).save(path+'/{}_pre.png'.format(i))
                    transforms.ToPILImage()(label_y[0,:,:,:]).save(path+'/{}_label.png'.format(i))
                        
        return psnr_mean.avg, ssim_mean.avg

def test_datasets(varPath, args, net, datasets, device, data):
    psnr_100 = {}
    ssim_100 = {}
    for dataset in datasets:
        print('testing dataset:{}...'.format(dataset))
        args.data_test = dataset
        loader_test, sampler_test = data.create_dataloader(args.test_data_path, test=True)

    # ######################################################################################################
        net.load_state_dict(remove_prefix(torch.load(varPath)['net']))
        psnr, ssim = test_snap_sp(net, loader_test, args, device, dataset = dataset, savefig=False)
        psnr_100[dataset] = psnr
        ssim_100[dataset] = ssim
    print_datasets_format(psnr_100, ssim_100)

def print_format(i, best_psnr, ssim):
    print('*'*10 + ' epoch:{} '.format(i+1) + '*'*10)
    print('*' + ' '*29 + '*')
    print('*' + ' '*8 + 'psnr:{:.4f}'.format(best_psnr) + ' '*9 + '*')
    print('*' + ' '*8 + 'ssim:{:.4f}'.format(ssim) + ' '*10 + '*')
    print('*' + ' '*29 + '*')
    print('*' + '*'*29 + '*')

def print_datasets_format(best_psnr, ssim):
    print('*'*50 + ' test results ' + '*'*46)
    print('*' + ' '*108 + '*')
    print('*' + ' '*11 + ' Set5 ' + ' '*14 + ' Set14 ' + ' '*15 + ' B100 ' + ' '*13 + ' Manga109 ' + ' '*11 + ' Urban100 ' + ' '*5 + '*')
    print(
        '*' + ' '*8 +
        '{:.2f}/{:.4f}'.format(best_psnr['Set5'],ssim['Set5']) +
        ' '*8 + '{:.2f}/{:.4f}'.format(best_psnr['Set14'],ssim['Set14']) +
        ' '*9 + '{:.2f}/{:.4f}'.format(best_psnr['B100'],ssim['B100']) +
        ' '*9 + '{:.2f}/{:.4f}'.format(best_psnr['Manga109'],ssim['Manga109']) +
        ' '*9 + '{:.2f}/{:.4f}'.format(best_psnr['Urban100'],ssim['Urban100']) +
        ' '*5 + '*'
        )
    print('*' + ' '*108 + '*')
    print('*' + '*'*108 + '*')

def test_single(varPath, args, net, img_path, device, scale):
    print('testing img:{}...'.format(img_path.split('/')[-1]))

# ######################################################################################################
    net.load_state_dict(remove_prefix(torch.load(varPath)['net']))
    net.eval()
    psnr_mean = AverageMeter('PSNR', ':2f')
    ssim_mean = AverageMeter('SSIM', ':4f')
    with torch.no_grad():
        hr = Image.open(img_path)
        # hr = hr.resize((720, 1280))
        hr,_,_ = hr.convert('YCbCr').split()
        hr = np.asarray(hr).astype(np.float32)
        lr = imresize_np(hr, 1 / scale, True)
        sr_bic = imresize_np(lr, scale, True)
        hr = Image.fromarray(hr)
        lr = Image.fromarray(lr)
        sr_bic = Image.fromarray(sr_bic)
        
        valid_y = transforms.PILToTensor()(lr).unsqueeze(0).to(device)
        label_y = transforms.PILToTensor()(hr).unsqueeze(0).to(device)
        sr_bic = transforms.PILToTensor()(sr_bic).unsqueeze(0).to(device)
        valid_y /= 255.
        label_y /= 255.
        sr_bic /= 255.

        # compute output
        meta = {'masks': [], 'general_mask':[], 'features':[], 'layer_features':[], 'tail_masks':[]}
        for _ in range(20):
            pre_y, meta = net((valid_y, meta))
        tik = time()
        pre_y, meta = net((valid_y, meta))
        cost_time = (time()-tik)*1000
        print('inference time:{} ms'.format(cost_time))
        pre_y = pre_y.clamp(0, 1)

        psnr = computer_psnr(pre_y.clamp(0,1), label_y, args.scale, data_range=1.)
        ssim = calc_ssim(pre_y, label_y, 1.)

        psnr_mean.update(psnr.item(), len(valid_y))
        ssim_mean.update(ssim.item(), len(valid_y))
    print('sr -> psnr:{:2f}\tssim:{:4f}'.format(psnr_mean.avg, ssim_mean.avg))
    psnr = computer_psnr(sr_bic.clamp(0,1), label_y, args.scale, data_range=1.)
    ssim = calc_ssim(sr_bic, label_y, 1.)
    print('bicubic -> psnr:{:2f}\tssim:{:4f}'.format(psnr, ssim))
    return cost_time, psnr, ssim

# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

# --------------------------------------------
# imresize for numpy image [0, 1]
# --------------------------------------------
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()

def test_bicubic(varPath, dataset, scale, device):
    # for dataset in datasets:
    print('testing dataset:{}...'.format(dataset))
    psnr_mean = AverageMeter('PSNR', ':2f')
    ssim_mean = AverageMeter('SSIM', ':4f')
    for path in sorted(os.listdir(varPath)):
        print(path)
        hr = Image.open(os.path.join(varPath, path))
        hr,hr_b,hr_r = hr.convert('YCbCr').split()
        hr = np.asarray(hr).astype(np.float32)
        lr = imresize_np(hr, 1 / scale, True)
        sr_bic = imresize_np(lr, scale, True)
        hr = Image.fromarray(hr)
        lr = Image.fromarray(lr)
        sr_bic = Image.fromarray(sr_bic)
        valid_y = transforms.PILToTensor()(lr).unsqueeze(0).to(device)
        label_y = transforms.PILToTensor()(hr).unsqueeze(0).to(device)
        sr_bic = transforms.PILToTensor()(sr_bic).unsqueeze(0).to(device)
        valid_y /= 255.
        label_y /= 255.
        sr_bic /= 255.
        psnr_bic = computer_psnr(sr_bic.clamp(0,1), label_y, args.scale, data_range=1.)
        ssim_bic = calc_ssim(sr_bic, label_y, 1.)
        print(psnr_bic, ssim_bic, len(valid_y))
        psnr_mean.update(psnr_bic.item(), len(valid_y))
        ssim_mean.update(ssim_bic.item(), len(valid_y))
    print(psnr_mean.avg, ssim_mean.avg)


def test_demo(varPath, args, net, hr, device, scale, cnt=0, savefig=False):
    # print('testing sequence:frame-{}...'.format(i))

# ######################################################################################################
    net.load_state_dict(remove_prefix(torch.load(varPath)['net']))
    net.eval()
    psnr_mean = AverageMeter('PSNR', ':2f')
    ssim_mean = AverageMeter('SSIM', ':4f')
    with torch.no_grad():
        hr,hr_b,hr_r = hr.convert('YCbCr').split()
        hr = np.asarray(hr).astype(np.float32)
        lr = imresize_np(hr, 1 / scale, True)
        sr_bic = imresize_np(lr, scale, True)
        hr = Image.fromarray(hr)
        lr = Image.fromarray(lr)
        sr_bic = Image.fromarray(sr_bic)
        
        valid_y = transforms.PILToTensor()(lr).unsqueeze(0).to(device)
        label_y = transforms.PILToTensor()(hr).unsqueeze(0).to(device)
        sr_bic = transforms.PILToTensor()(sr_bic).unsqueeze(0).to(device)
        valid_y /= 255.
        label_y /= 255.
        sr_bic /= 255.

        # compute output
        # meta = {'masks': [], 'general_mask':[], 'features':[], 'layer_features':[], 'tail_masks':[]}
        for _ in range(10):
            meta = {'masks': [], 'general_mask':[], 'features':[], 'layer_features':[], 'tail_masks':[]}
            pre_y, meta = net((valid_y, meta))
        meta = {'masks': [], 'general_mask':[], 'features':[], 'layer_features':[], 'tail_masks':[]}
        tik = time()
        pre_y, meta = net((valid_y, meta))
        cost_time = (time()-tik)*1000
        # print('inference time:{} ms'.format(cost_time))
        pre_y = pre_y.clamp(0, 1)

        psnr = computer_psnr(pre_y.clamp(0,1), label_y, args.scale, data_range=1.)
        ssim = calc_ssim(pre_y, label_y, 1.)

        psnr_mean.update(psnr.item(), len(valid_y))
        ssim_mean.update(ssim.item(), len(valid_y))
    psnr_sr = psnr_mean.avg
    ssim_sr = ssim_mean.avg
    psnr_bic = computer_psnr(sr_bic.clamp(0,1), label_y, args.scale, data_range=1.)
    ssim_bic = calc_ssim(sr_bic, label_y, 1.)
    if savefig:
            former = './output'
            out_path_hr = os.path.join(former, 'HR')
            out_path_sr = os.path.join(former, 'SR', 'X{}'.format(scale))
            out_path_bic = os.path.join(former, 'SR_bic', 'X{}'.format(scale))
            image = transforms.ToPILImage()(label_y.squeeze(0))
            image = Image.merge('YCbCr', (image, hr_b, hr_r)).convert('RGB')
            
            if os.path.exists(out_path_hr):
                image.save(os.path.join(out_path_hr, 'img{:05}.png'.format(cnt+1)))
            else:
                os.makedirs(out_path_hr)
                image.save(os.path.join(out_path_hr, 'img{:05}.png'.format(cnt+1)))

            image = transforms.ToPILImage()(pre_y.squeeze(0))
            image = Image.merge('YCbCr', (image, hr_b, hr_r)).convert('RGB')
            if os.path.exists(out_path_sr):
                image.save(os.path.join(out_path_sr, 'img{:05}x{}.png'.format(cnt+1, scale)))
            else:
                os.makedirs(out_path_sr)
                image.save(os.path.join(out_path_sr, 'img{:05}x{}.png'.format(cnt+1, scale)))

            image = transforms.ToPILImage()(sr_bic.squeeze(0))
            image = Image.merge('YCbCr', (image, hr_b, hr_r)).convert('RGB')
            if os.path.exists(out_path_bic):
                image.save(os.path.join(out_path_bic, 'img{:05}x{}.png'.format(cnt+1, scale)))
            else:
                os.makedirs(out_path_bic)
                image.save(os.path.join(out_path_bic, 'img{:05}x{}.png'.format(cnt+1, scale)))

    return cost_time, psnr_sr, ssim_sr, psnr_bic, ssim_bic

def test_big_data(path, size, scale):
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(args.log_dir, "log.log"), "a+")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'
    varPath = args.test_model_path
    net = SRGFS_Inf(args.n_feats, args.scale, args.n_resblocks, args.alpha).to(device)
    params = float(net.params()) / 1e3
    net.eval()

    h,w,c = (int(alpha) for alpha in size.split('x'))
    times = AverageMeter('SSIM', ':4f')
    psnrs_sr = AverageMeter('PSNR', ':2f')
    ssims_sr = AverageMeter('SSIM', ':4f')
    psnrs_bic = AverageMeter('PSNR', ':2f')
    ssims_bic = AverageMeter('SSIM', ':4f')
    with open(path,'rb') as fp:
        i = 0
        while(1):
            data_bytes = fp.read(h*w*c)
            if data_bytes == 0 or data_bytes == b'':
                break
            data = np.reshape(np.frombuffer(data_bytes,'B'),(h,w,c))

            image = Image.fromarray(data)
            cost_time, psnr_sr, ssim_sr, psnr_bic, ssim_bic = test_demo(varPath, args, net, image, device, scale, i, savefig=args.savefig)
            times.update(cost_time)
            psnrs_sr.update(psnr_sr)
            ssims_sr.update(ssim_sr)
            psnrs_bic.update(psnr_bic)
            ssims_bic.update(ssim_bic)
            image = imresize_np(data, 1 / scale, True)
            image = Image.fromarray(image.clip(0,255).astype(np.uint8))
            logger.info('processed img-{:06}'.format(i+1))
            i += 1
            logger.debug('cost_time:{:4f} ms\tpsnr:{:2f} db\tssim:{:4f} \tpsnr_bic:{:2f} db\tssim_bic:{:4f} '.format(times.avg, psnrs_sr.avg, ssims_sr.avg, psnrs_bic.avg, ssims_bic.avg))

    logger.info('#'*25)
    logger.info('cost_time:{:4f}ms\tparameters:{:2f}K\tpsnr:{:2f}db\tssim:{:4f}\tpsnr_bic:{:2f}db\tssim_bic:{:4f} '.format(times.avg, params, psnrs_sr.avg, ssims_sr.avg, psnrs_bic.avg, ssims_bic.avg))
    logger.info('#'*25)


def main():
# ###########################################################################################
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    logger = Logger(args, False)
    data = Data(args, logger, 0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'
    varPath = args.test_model_path
    net = SRGFS_Inf(args.n_feats, args.scale, args.n_resblocks, args.alpha).to(device)
    net.eval()

    datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K']
    if args.test_type == 'benchmark':
# ================================= test benchmark ================================= #
        test_datasets(varPath, args, net, datasets, device, data)
    elif args.test_type == 'single':
# ================================= test single image ================================= #
        test_single(varPath, args, net, args.test_data_path, device, args.scale)
    elif args.test_type == 'big_data':
        test_big_data(args.test_data_path, args.super_resolution, args.scale)
    elif args.test_type == 'bicubic':
        test_bicubic(args.test_data_path, args.data_test, args.scale, device)
    else:
        assert print('No Implementation')
    
    
if __name__ == '__main__':
    main()
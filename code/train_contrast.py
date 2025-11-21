import argparse
import random
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import contrast_transform
import utils
from scipy import ndimage
import numpy as np
from networks.res_50 import Model
from torchvision import transforms
from scipy.ndimage import zoom
from dataloaders.dataset import BaseDataSets, RandomGenerator

def convert_RGB(image_tensor):
    rgb_images = []
    for image in image_tensor:
        if image.shape[0] == 1:
            rgb_image = image.repeat(3, 1, 1)
            rgb_images.append(rgb_image)
        else:
            rgb_images.append(image)
    return torch.stack(rgb_images)

def contrastive_transform1_tensor(image_tensor):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    transformed_images = []
    for image in image_tensor:
        # Convert to RGB if the image has only one channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        pil_image = to_pil(image)
        transformed_image = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])(pil_image)

        tensor_image = to_tensor(transformed_image)
        transformed_images.append(tensor_image)

    return torch.stack(transformed_images)


def contrastive_transform2_tensor(image_tensor):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    transformed_images = []
    for image in image_tensor:
        # Convert to RGB if the image has only one channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        pil_image = to_pil(image)
        transformed_image = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        ])(pil_image)

        tensor_image = to_tensor(transformed_image)
        transformed_images.append(tensor_image)

    return torch.stack(transformed_images)

def test_image_label(image, label, patch_size=[256, 216]):
    image_numpy = image[0].cpu().detach().numpy()
    label_numpy = label[0].cpu().detach().numpy()

    if len(image_numpy.shape) == 3:
        # Process multi-slice (3D) image
        input_tensors = []
        for ind in range(image_numpy.shape[0]):
            slice = image_numpy[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            # Convert to 3-channel RGB
            rgb_slice = torch.from_numpy(slice).unsqueeze(0).repeat(3, 1, 1)
            input_tensor = rgb_slice.unsqueeze(0).float().cuda()
            input_tensors.append(input_tensor)
        input_tensors = torch.cat(input_tensors, dim=0)  # Concatenate tensors along batch dimension
    else:
        # Process single-slice (2D) image
        # Convert to 3-channel RGB
        rgb_image = torch.from_numpy(image_numpy).unsqueeze(0).repeat(3, 1, 1)
        input_tensor = rgb_image.unsqueeze(0).float().cuda()
        input_tensors = input_tensor

    label_tensor = torch.from_numpy(label_numpy).unsqueeze(0).float().cuda()

    return input_tensors, label_tensor



# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for sampled_batch in train_bar:
        image, label = sampled_batch['image'],sampled_batch['label']
        image, label = image.cuda(), label.cuda()
        image_q = contrastive_transform1_tensor(image)
        image_k = contrastive_transform2_tensor(image)
        image_q = image_q.cuda()
        image_k = image_k.cuda()
        _, query = encoder_q(image_q)

        # shuffle BN
        idx = torch.randperm(image_k.size(0), device=image_k.device)
        _, key = encoder_k(image_k[idx])
        key = key[torch.argsort(idx)]

        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(image_q.size(0), dtype=torch.long, device=image_q.device))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += image_q.size(0)
        total_loss += loss.item() * image_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for sample_test in tqdm(memory_data_loader, desc='Feature extracting'):
            sample_test['image'] = convert_RGB(sample_test['image'])
            feature, out = net(sample_test['image'].cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = memory_data_loader.dataset.get_labels()

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for sbatch in test_bar:
            sbatch['image'], sbatch['label'] = test_image_label(sbatch['image'], sbatch['label'])
            feature, out = net(sbatch['image'])

            total_num += sbatch['image'].size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(sbatch['image'].size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(sbatch['image'].size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(sbatch['image'].size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == sbatch['label'].unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == sbatch['label'].unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--root_path', type=str, default=r'G:\cardiac_data\WSL4MIS-main\data\ACDC',help='Name of Experiment')
    parser.add_argument('--fold', type=str, default='fold1', help='cross validation')
    parser.add_argument('--sup_type', type=str, default='scribble', help='supervision type')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=2, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--patch_size', type=list, default=[256, 256],help='patch size of network input')

    # args parse
    args = parser.parse_args()
    feature_dim, m, temperature, momentum = args.feature_dim, args.m, args.temperature, args.momentum
    k, batch_size, epochs = args.k, args.batch_size, args.epochs

    # data prepare
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_memory = BaseDataSets(base_dir=args.root_path, split="train",  fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    memoryloader = DataLoader(db_train, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # model setup and optimizer config
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()
    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    optimizer = optim.Adam(model_q.parameters(), lr=1e-3, weight_decay=1e-6)

    # c as num of train class
    c = 4
    # init memory queue as unit random vector ---> [M, D]
    memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size, epochs)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model_q, model_k, trainloader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model_q, memoryloader, valloader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        #save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_results.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model_q.state_dict(), 'results/{}_model.pth'.format(save_name_pre))

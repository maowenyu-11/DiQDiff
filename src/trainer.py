import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
from Modules import KMeans
import pickle
import torch.nn.functional as F
import os
import time as Time

def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def LSHT_inference(model, args, data_loader):
    device = args.device
    model = model.to(device)
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            
            rep_diffu= model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = model.diffu_rep_pre(rep_diffu)
            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)




# def contra_loss(diffu,labels,num_cluster,criterion):
#     centers=diffu[torch.randperm(len(diffu))[:num_cluster]]
#     for i in range(num_cluster):
#         if torch.sum(labels == i) == 0:
#             centers[i] = diffu[torch.randint(0, diffu.size(0), (1,))]
#         else:
#             centers[i] = torch.mean(diffu[labels == i], dim=0)
    
#     distance=None    
#     # centers=torch.stack(centers)
#     for i in range(num_cluster):
#         centers = torch.cat((centers[i:i + 1], centers[0:i],
#         centers[i + 1:]), 0)
#         distance_= torch.einsum('nc,kc->nk', [
#     nn.functional.normalize(diffu[labels==i], dim=1),
#     nn.functional.normalize(centers, dim=1)
# ])

#         if distance is None:
#             distance = F.softmax(distance_, dim=1)
#         else:
#             distance = torch.cat((distance, F.softmax(distance_, dim=1)),
#                                             0)
#             idx = torch.zeros(distance.shape[0], dtype=torch.long).cuda()

#     loss = criterion(distance, idx)
#     return loss

def model_train(tra_data_loader, val_data_loader, test_data_loader, model, args, logger):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model = model.to(device)
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model = nn.DataParallel(model)
    optimizer = optimizers(model, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0, 'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0
    # criterion = nn.CrossEntropyLoss().to(args.device)
    for epoch_temp in range(epochs):        
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model.train()
        metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        flag_update = 0
        train_start = Time.time()
        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            diffu_rep,code= model(train_batch[0], train_batch[1], train_flag=True)  
            loss_diffu_value,loss_contra= model.loss_diffu_ce(diffu_rep, train_batch[1])  ## use this not above
            # loss_contra=contra_loss(diffu_rep)
            # loss_contra=0
            loss_all = loss_diffu_value+args.lambda_contra*loss_contra
            loss_all.backward()
        
            optimizer.step()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f Loss_contra: %.4f' % (index_temp, len(tra_data_loader), loss_all.item(),loss_contra))
        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))
        print("Train cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-train_start)))

        lr_scheduler.step()



        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
                # metrics_dict_mean = {}
                eval_start = Time.time()

                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    rep_diffu,code= model(val_batch[0], val_batch[1], train_flag=False)

                    scores_rec_diffu = model.diffu_rep_pre(rep_diffu)    ### inner_production
                    

                    metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
            # if epoch_temp % 10 == 0:
            #   torch.save(rep_diffu, str(epoch_temp)+'generation_nv.pt')            
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
            # if epoch_temp % 10 == 0:
            #    torch.save(code, str(epoch_temp)+'code_nc.pt')          
            print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
        
            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                best_model = copy.deepcopy(model)
            if bad_count >= args.patience:
                
                break
            
    
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
        
    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model)
    
    
    top_100_item = []
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            rep_diffu,code= best_model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)   ### Inner Production
            # scores_rec_diffu = best_model.routing_rep_pre(rep_diffu)   ### routing
            
            _, indices = torch.topk(scores_rec_diffu, k=100)
            top_100_item.append(indices)

            metrics = hrs_and_ndcgs_k(scores_rec_diffu, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    print(args)

    if args.diversity_measure:
        path_data = '../datasets/data/category/' + args.dataset +'/id_category_dict.pkl'
        with open(path_data, 'rb') as f:
            id_category_dict = pickle.load(f)
        id_top_100 = torch.cat(top_100_item, dim=0).tolist()
        category_list_100 = []
        for id_top_100_temp in id_top_100:
            category_temp_list = [] 
            for id_temp in id_top_100_temp:
                category_temp_list.append(id_category_dict[id_temp])
            category_list_100.append(category_temp_list)
        category_list_100.append(category_list_100)
        path_data_category = '../datasets/data/category/' + args.dataset +'/DiffuRec_top100_category.pkl'
        with open(path_data_category, 'wb') as f:
            pickle.dump(category_list_100, f)
            

    return best_model, test_metrics_dict_mean
    

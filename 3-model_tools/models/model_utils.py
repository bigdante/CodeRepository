import os.path

import torch


def total_paramters(model):
    '''
    calculate the total parameters in model, the unit is M
    :param model:
    :return:
    '''
    # print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: {} ".format(total))
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    return total


def save_model(model, only_params=False, path="./model", ckpt_name="save.pt", ckpt_dict=None):
    '''
    save model
    :param ckpt_dict: user's definition ckpt
        example:
        #saving a checkpoint assuming the network class named ClassNet
        checkpoint={'modle':ClassNet(),
                     'model_state_dict':model.state_dict(),
                     'optimize_state_dict':optimizer.state_dict(),
                     'epoch':epoch}
        torch.save(checkpoint,'checkpoint.pkl')
    :param model:
    :param only_params:
    :param path:
    :param ckpt_name:
    :return:
    '''
    save_path = os.path.join(path, ckpt_name)
    if not os.path.exists(path):
        os.mkdir(path)
    if ckpt_dict:
        torch.save(ckpt_dict, 'checkpoint.pkl')
    elif only_params:
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model, save_path)


def load_model(path, model=None, optimizer=None, only_params=False, from_pkl_dict=False):
    '''
    load model
    :param optimizer:
    :param from_pkl_dict: from pkl to load model
    :param path:
    :param model: if noly_parms is true, we need a model
    :param only_params:
    :return:
    '''
    if from_pkl_dict:
        checkpoint = torch.load(path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if only_params:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = torch.load(path)
    return model

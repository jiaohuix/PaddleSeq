import os
import paddle
import shutil
import numpy as np
from collections import OrderedDict


def save_model(conf,model, optimizer,save_dir,nbest=5):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    paddle.save(model.state_dict(), os.path.join(save_dir, "model.pdparams"))
    paddle.save(optimizer.state_dict(), os.path.join(save_dir, "model.pdopt"))
    # save args
    args_dict = OrderedDict()
    args_dict["conf"] = conf
    paddle.save(args_dict,os.path.join(save_dir, "model.args"))


    ''' save n best and drop old best'''
    if save_dir.find('best')!=-1:
        base_dir=os.path.dirname(save_dir)
        all_names=os.listdir(base_dir)
        ckpt_names=[name for name in all_names if os.path.isdir(os.path.join(base_dir,name)) and name.find('model')!=-1]
        best_names=[name for name in ckpt_names if name.find('model_best_')!=-1]
        best_names=list(sorted(best_names,key=lambda name: float(name.replace('model_best_',''))))
        if len(best_names)>nbest:
            print("removing: {}".format(os.path.join(base_dir, best_names[0])))
            shutil.rmtree(os.path.join(base_dir,best_names[0]))
            # print(os.path.exists(os.path.join(base_dir, best_names[0])))
    ckpt_ls=os.listdir(os.path.dirname(save_dir))
    print(f"current checkpoints: {ckpt_ls}")



def set_freeze_by_names(model,exclude_layers,freeze=True):
    for name, layer in model.named_children():
        if name in exclude_layers:
            continue
        for param in layer.parameters():
            param.stop_gradient = freeze


def freeze_by_names(model,exclude_layers):
    set_freeze_by_names(model,exclude_layers,freeze=True)
    print(f"model parameters have been frozen, exclude {exclude_layers}.")

def unfreeze_by_names(model,exclude_layers):
    set_freeze_by_names(model,exclude_layers,freeze=False)
    print(f"model parameters have been unfrozen.")

def save_embedding(embed_dict_path, vocab, embedding):
    '''
    embed_dict_path: folder  # folder/vocab.npy
    eg: save src share vocab   save_embedding("path", model.src_vocab, model.encoder.embed_tokens)
    '''
    embed_dict = {"vocab_size": len(vocab), "embed_dim": embedding.weight.shape[1],
                  "token2id": {}, "embedding": embedding.weight.numpy()}
    for idx in range(len(vocab)):
        token = vocab.to_tokens(idx)
        embed_dict["token2id"][token] = idx
    if not os.path.exists(embed_dict_path):
        os.makedirs(embed_dict_path)
    path = os.path.join(embed_dict_path, "vocab.npy")
    np.save(path, [embed_dict])
    print(f"save embedding to {path} success.")

def load_embedding(embed_dict_path, vocab, embedding):
    '''
    eg: load tgt embed
        load_embedding("path",model.tgt_vocab, model.decoder.embed_tokens)
    '''
    assert len(vocab) == embedding.weight.shape[0], "vocab size should match embed_nums."
    path = os.path.join(embed_dict_path, "vocab.npy")
    embed_dict = np.load(path, allow_pickle=True)[0]
    weights = []
    num=0
    for idx in range(len(vocab)):
        token = vocab.to_tokens(idx)
        if token in embed_dict["token2id"]:
            embed_idx = embed_dict["token2id"][token]
            embed = paddle.to_tensor(embed_dict["embedding"][embed_idx], dtype="float32")
            weights.append(embed)
        else:
            weights.append(embedding.weight[idx])
            num+=1
    weights = paddle.stack(weights, axis=0)
    embedding.weight.set_value(weights)
    print(f"load embed form {embed_dict_path} success.")
    return embedding
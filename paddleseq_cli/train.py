import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))
import time
import math
import paddle
import paddle.distributed as dist
from paddlenlp.transformers import  LinearDecayWithWarmup
from paddle.optimizer.lr import CosineAnnealingDecay, NoamDecay
from visualdl import LogWriter
from paddleseq_cli.config import get_arguments, get_config
from paddleseq.reader import prep_dataset, prep_loader
from paddleseq_cli.validate import validation
from paddleseq.models import build_model
from paddleseq.criterions import CrossEntropyCriterionBase
from paddleseq.lr_scheduler import ReduceOnPlateauWithAnnael,InverseSquareRoot,KneeLRScheduler
from paddleseq.utils import set_paddle_seed,get_grad_norm
from paddleseq.logging import NMTMetric,get_logger
from paddleseq.checkpoint_utils import save_model

def train_one_epoch(conf,
                    dataloader,
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    epoch,
                    step_id,
                    metric,
                    logger,
                    logwriter,
                    max_epoch,
                    pad_idx=1,
                    amp=False,
                    log_steps=100,
                    update_freq=1,
                    scheduler=None):  # for warmup
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nmt model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        log_steps: int, num of iters to log info
        update_freq: int, num of iters for accumulating gradients
    Returns:
        train_loss_meter.avg
        train_acc_meter.avg
        train_time
    """

    model.train()
    # Train loop
    gnorm = 0  # gradient norm
    sentences = 0
    tic_train = time.time()
    for batch_id, input_data in enumerate(dataloader):
        (samples_id, src_tokens, prev_tokens, tgt_tokens) = input_data
        sample = {"src_tokens":src_tokens,"prev_tokens":prev_tokens,"tgt_tokens":tgt_tokens}
        # for mixed precision training
        if amp is True:
            # with paddle.amp.auto_cast():
            with paddle.amp.auto_cast(
                    enable=True ,
                    custom_white_list=["layer_norm", "softmax", "gelu"],
                    custom_black_list=[
                        "reduce_sum", "c_softmax_with_cross_entropy",
                        "c_embedding"
                    ]):
                logits, sum_cost, avg_cost, token_num = criterion(model, sample)
                loss = avg_cost
                scaled = scaler.scale(loss)
                scaled.backward()
            gnorm = get_grad_norm(grads=[p.grad for p in optimizer._param_groups])
            if ((batch_id + 1) % update_freq == 0) or (batch_id + 1 == len(dataloader)):
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
        # for full precision training
        else:
            logits, sum_cost, avg_cost, token_num = criterion(model, sample)
            loss = avg_cost
            loss.backward()
            gnorm = get_grad_norm(grads=[p.grad for p in optimizer._param_groups])
            if ((batch_id + 1) % update_freq == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.clear_grad()

        # aggregate metric
        loss, nll_loss, ppl = metric.update(sum_cost, logits, target=tgt_tokens,
                                            sample_size=token_num, pad_id=pad_idx,gnorm=gnorm)
        sentences += src_tokens.shape[0]

        # log
        if (batch_id + 1) % log_steps == 0:
            avg_bsz = sentences / (batch_id + 1)
            bsz = round(avg_bsz * update_freq * dist.get_world_size() ,1)
            avg_total_steps = int(len(dataloader.dataset) // dist.get_world_size() // avg_bsz )  # Number of iterations of each epoch in a single card
            cur_steps=avg_total_steps * (epoch-1) + batch_id + 1 # current forward steps (single card)
            # num_updates = (cur_steps//update_freq) *  dist.get_world_size()
            num_updates = (cur_steps//update_freq)
            loss, nll_loss, ppl, gnorm = metric.accumulate()

            logger.info(
                f"Train| epoch:[{epoch}/{int(max_epoch)}], step:[{batch_id + 1}/{avg_total_steps}], "
                f"speed:{log_steps / (time.time() - tic_train):.2f} step/s, "
                f"loss:{float(loss):.3f}, nll_loss:{float(nll_loss):.3f}, ppl:{float(ppl):.2f}, bsz:{bsz}, "
                f"gnorm:{gnorm:.3f}, num_updates:{num_updates}, lr:{optimizer.get_lr():.9f}")
            tic_train = time.time()

        if dist.get_rank() == 0:
            logwriter.add_scalar(tag="train/loss", step=step_id, value=loss)
            logwriter.add_scalar(tag="train/ppl", step=step_id, value=ppl)
            if not conf.train.amp: # if use amp,gnorm will explode
                logwriter.add_scalar(tag="train/gnorm", step=step_id, value=gnorm)

        # save model after several steps
        if (conf.train.save_step>0) and (step_id % conf.train.save_step == 0) and (dist.get_rank() == 0):
            save_model(conf, model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, f"step_{step_id}"))

        if isinstance(scheduler,
                      (LinearDecayWithWarmup, CosineAnnealingDecay, NoamDecay,InverseSquareRoot,KneeLRScheduler)):  # these scheds  updated each step
            scheduler.step(step_id)
        step_id += 1


    return step_id, gnorm


def get_optimizer_scheduler(conf,model,dataloader,global_step_id):
    scheduler = None
    learn_args=conf.learning_strategy
    sched_args =conf.learning_strategy.scheduler[learn_args.sched]
    from paddlenlp.transformers  import LinearDecayWithWarmup
    if learn_args.sched == "plateau":
        scheduler = ReduceOnPlateauWithAnnael(learning_rate=float(learn_args.learning_rate),
                                              patience=sched_args.patience,
                                              force_anneal=sched_args.force_anneal,
                                              factor=sched_args.lr_shrink,
                                              min_lr=learn_args.min_lr)  # reduce the learning rate until it falls below 10−4
    elif learn_args.sched == "cosine":
        scheduler = CosineAnnealingDecay(learning_rate=float(learn_args.learning_rate),
                                         T_max=sched_args.t_max,
                                         last_epoch=global_step_id)

    elif learn_args.sched == "linear":
        scheduler = LinearDecayWithWarmup(learning_rate=float(learn_args.learning_rate),
                                          warmup=learn_args.warm_steps,
                                          last_epoch=global_step_id if conf.train.resume else -1,
                                          total_steps=conf.train.max_epoch * len(dataloader))

    elif learn_args.sched == "noamdecay":
        scheduler = NoamDecay(d_model=sched_args.d_model,
                              warmup_steps=learn_args.warm_steps,
                              learning_rate=float(learn_args.learning_rate),
                              last_epoch=global_step_id)
    elif learn_args.sched == "inverse_sqrt":
        scheduler = InverseSquareRoot(warmup_init_lr=float(sched_args.warmup_init_lr),
                                      warmup_steps=learn_args.warm_steps,
                                      learning_rate=float(learn_args.learning_rate),
                                      last_epoch=global_step_id)

    elif learn_args.sched == "knee":
        epoch_steps = len(dataloader)
        explore_steps =  sched_args.explore_epochs * epoch_steps
        total_steps = conf.train.max_epoch * epoch_steps
        print("explore_steps",explore_steps,"total_steps",total_steps)
        scheduler = KneeLRScheduler(warmup_init_lr=float(sched_args.warmup_init_lr),
                                    peak_lr=float(learn_args.learning_rate),
                                    warmup_steps=learn_args.warm_steps,
                                    explore_steps=explore_steps,
                                    total_steps= total_steps,
                                    last_epoch=global_step_id if conf.train.resume else -1)


    assert scheduler is not None, "Sched should in [plateau|cosine|linear|noamdecay|inverse_sqrt|knee]."
    assert learn_args.clip_type in ["local","global"], "clip_type should in [local|global]."
    if learn_args.clip_norm<=0:
        clip=None
    else:
        clip_map={"local":"ClipGradByNorm","global":"ClipGradByGlobalNorm"}
        clip=getattr(paddle.nn,clip_map[learn_args.clip_type])(clip_norm=learn_args.clip_norm)

    optimizer = None
    if learn_args.optim == "nag":
        optim_args=conf.learning_strategy.optimizer.nag
        optimizer = paddle.optimizer.Momentum(
            learning_rate=scheduler,
            weight_decay=float(learn_args.weight_decay),
            grad_clip=clip,
            parameters = model.parameters(),
            momentum=optim_args.momentum,
            use_nesterov=optim_args.use_nesterov)
    elif learn_args.optim == "adam":
        optim_args=conf.learning_strategy.optimizer.adam
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler,
            weight_decay=float(learn_args.weight_decay),
            grad_clip=clip,
            parameters=model.parameters(),
            beta1=optim_args.beta1,
            beta2=optim_args.beta2)
    elif learn_args.optim == "adamw":
        optim_args=conf.learning_strategy.optimizer.adam
        optimizer = paddle.optimizer.AdamW(
            learning_rate=scheduler,
            weight_decay=float(learn_args.weight_decay),
            grad_clip=clip,
            parameters=model.parameters(),
            beta1=optim_args.beta1,
            beta2=optim_args.beta2)
    assert optimizer is not None, "Optimizer should in [nag|adam|adamw]"

    return optimizer,scheduler

def early_stop(conf,optimizer,val_loss,lowest_val_loss,num_runs,gnorm,global_step_id,logger):
    stop_flag=False
    # 1.stop training when lr too small
    cur_lr = round(optimizer.get_lr(), 5)
    min_lr = round(conf.learning_strategy.min_lr, 5)
    if (cur_lr <= min_lr):
        logger.info(f"early stop since min lr has reached.")
        stop_flag=True

    # 2.early stop for patience
    if conf.train.stop_patience > 1:
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            num_runs = 0
        else:
            num_runs += 1
            if num_runs >= conf.train.stop_patience:
                logger.info(f"early stop since valid performance hasn't improved for last {conf.train.early_stop_num} runs")
                stop_flag=True

    # 3.early stop for gradient
    # if float(gnorm) >= float("inf") or math.isnan(float(gnorm)):
    #     logger.info(f"early stop since grdient norm is inf.")
    #     stop_flag=True

    # 4.early stop for max_update
    if (global_step_id > conf.train.max_update) and (conf.train.max_update!=-1):
        stop_flag=True

    return stop_flag,lowest_val_loss,num_runs

def main_worker(*args):
    # 0.Preparation
    conf = args[0]
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    if world_size>1:
        dist.init_parallel_env()
    last_epoch = conf.train.last_epoch
    logger = get_logger(loggername=f"{conf.model.model_name}_rank{local_rank}", save_path=conf.SAVE)
    logger.info(f'----- world_size = {world_size}, local_rank = {local_rank}')
    seed = conf.seed + local_rank
    set_paddle_seed(seed)
    tic=time.time()
    # 1. Create train and val dataset
    dataset_train, dataset_val = args[1], args[2]
    # Create training dataloader
    train_loader = None
    if not conf.eval:
        train_loader = prep_loader(conf, dataset_train, mode="train" ,multi_process=world_size>1)
        logger.info(f"----- Total of train set:{len(train_loader.dataset)} ,train batch: {len(train_loader)} [single gpu]")
    dev_loader = prep_loader(conf, dataset_val, mode="dev",multi_process=world_size>1)
    toc=time.time()
    logger.info(f"----- Total of valid set:{len(dev_loader.dataset)} ,valid batch: {len(dev_loader)} [single gpu]")
    logger.info(f"Load data cost {toc-tic} seconds.")
    if local_rank == 0:
        logger.info(f"configs:\n{conf}")

    # 2. Create model
    model = build_model(conf, is_test=False)
    if world_size>1:
        model = paddle.DataParallel(model)
    if local_rank == 0:
        logger.info(f"model:\n{model}")

    # 3. Define criterion
    criterion = CrossEntropyCriterionBase(conf.learning_strategy.label_smooth_eps, pad_idx=conf.model.pad_idx)
    metric = NMTMetric(name=conf.model.model_name)
    logwriter = None
    best_bleu = 0
    if local_rank == 0:
        strtime = time.strftime("Time%m%d_%Hh%Mm", time.localtime(time.time()))
        lang_direction=f"{conf.data.src_lang}{conf.data.tgt_lang}"
        logdir=os.path.join(conf.SAVE,f"vislogs/{lang_direction}/{conf.model.model_name}_{strtime}")
        logwriter=LogWriter(logdir=logdir)

    # 4. Define optimizer and lr_scheduler
    # global_step_id = conf.train.last_epoch * len(train_loader) + 1 if train_loader is not None else 0
    global_step_id = conf.train.last_step
    optimizer,scheduler=get_optimizer_scheduler(conf,model,train_loader,global_step_id)

    # 5. Load  resume  optimizer states
    if conf.train.resume:
        model_path = os.path.join(conf.train.resume, "model.pdparams")
        optim_path = os.path.join(conf.train.resume, 'model.pdopt')
        assert os.path.isfile(model_path) is True, f"File {model_path} does not exist."
        assert os.path.isfile(optim_path) is True, f"File {optim_path} does not exist."
        model_state = paddle.load(model_path)
        opt_state = paddle.load(optim_path)
        if conf.learning_strategy.reset_lr:  # weather to reset lr
            opt_state["LR_Scheduler"]["last_lr"] = conf.learning_strategy.learning_rate
        # resume best bleu
        # best_bleu = opt_state['LR_Scheduler'].get('best_bleu', 0)
        model.set_dict(model_state)
        optimizer.set_state_dict(opt_state)
        logger.info(
            f"----- Resume Training: Load model and optmizer states from {conf.train.resume},LR={optimizer.get_lr():.5f}----- ")

    # 6. Validation
    if conf.eval:
        logger.info("----- Start Validating")
        val_loss, val_nll_loss, val_ppl, dev_bleu = validation(conf, dev_loader, model, criterion, logger)
        return

    # 6. Start training and validation
    # def GradScaler
    scale_init = conf.train.fp16_init_scale
    growth_interval = conf.train.growth_interval if conf.train.amp_scale_window else 2000
    scaler = paddle.amp.GradScaler(init_loss_scaling=scale_init, incr_every_n_steps=growth_interval)
    lowest_val_loss = 0
    num_runs = 0
    for epoch in range(last_epoch + 1, conf.train.max_epoch + 1):
        # train
        logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.5f}")
        global_step_id,gnorm = train_one_epoch(
            conf,
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step_id=global_step_id,
            metric=metric,
            logger=logger,
            logwriter=logwriter,
            max_epoch=conf.train.max_epoch,
            pad_idx=conf.model.pad_idx,
            amp=conf.train.amp,
            log_steps=conf.train.log_steps,
            update_freq=conf.train.update_freq,
            scheduler=scheduler
        )
        metric.reset()
        # evaluate model on valid data after one epoch
        val_loss, val_nll_loss, val_ppl, dev_bleu = validation(conf, dev_loader, model, criterion, logger)

        # update config
        conf.defrost()
        conf.train.last_epoch = epoch
        conf.train.last_step = global_step_id
        conf.freeze()
        # save best model state
        if (best_bleu < dev_bleu) and (local_rank == 0):
            best_bleu = dev_bleu
            save_dir = os.path.join(conf.SAVE, conf.model.save_model, f"model_best_{best_bleu}")
            save_model(conf, model, optimizer, save_dir=save_dir)
            logger.info(f"Epoch:[{epoch}] | Best Valid Bleu: [{best_bleu:.3f}] saved to {save_dir}!")
        elif (local_rank==0):
            save_dir = os.path.join(conf.SAVE, conf.model.save_model, f"model_best_{dev_bleu}")
            save_model(conf, model, optimizer, save_dir=save_dir)
            logger.info(f"Epoch:[{epoch}] | Although not the highest valid bleu：[{dev_bleu:.3f}], save to replace the lowest ckpt.")

        # visualize valid metrics
        if local_rank == 0:
            logwriter.add_scalar(tag="valid/loss", step=epoch, value=val_loss)
            logwriter.add_scalar(tag="valid/ppl", step=epoch, value=val_ppl)
            logwriter.add_scalar(tag="valid/bleu", step=epoch, value=dev_bleu)


        # adjust learning rate when val ppl stops improving (each epoch).
        if conf.learning_strategy.sched == "plateau":
            scheduler.step(val_ppl)

        # early stop
        stop_flag,lowest_val_loss,num_runs=early_stop(conf,optimizer,val_loss ,lowest_val_loss,num_runs,gnorm,global_step_id,logger)
        if stop_flag:break

        if (epoch % conf.train.save_epoch == 0) and (local_rank == 0):
            save_model(conf, model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, f"epoch_{epoch}"))

    # save last model
    if (conf.model.save_model) and (local_rank == 0):
        save_model(conf, model, optimizer, save_dir=os.path.join(conf.SAVE, conf.model.save_model, "epoch_final"))

    if local_rank == 0:
        logwriter.close()


def main():
    args = get_arguments()
    conf = get_config(args)
    if not conf.eval:
        dataset_train = prep_dataset(conf, mode="train")
    else:
        dataset_train = None
    dataset_dev = prep_dataset(conf, mode="dev")

    dist.spawn(main_worker, args=(conf, dataset_train, dataset_dev,), nprocs=conf.ngpus)


if __name__ == "__main__":
    main()

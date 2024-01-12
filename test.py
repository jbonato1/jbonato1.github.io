
args.optim = 'sgd'
args.gamma = 0.99
args.alpha = 0.001
args.beta = 0
args.smoothing = 0.0
args.msteps = 2
args.clip = 0.2
args.sstart = 10
args.kd_T = 4
args.distill = 'kd'

args.sgda_batch_size = 128
args.del_batch_size = 32
args.sgda_epochs = 3
args.sgda_learning_rate = 0.0005
args.lr_decay_epochs = [3,5,9]
args.lr_decay_rate = 0.1
args.sgda_weight_decay = 5e-4
args.sgda_momentum = 0.9

#this is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
#For SGDA smoothing
beta = 0.1
def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
    1 - beta) * averaged_model_parameter + beta * model_parameter
swa_model = torch.optim.swa_utils.AveragedModel(
    model_s, avg_fn=avg_fn)


acc_rs = []
acc_fs = []
acc_ts = []
acc_vs = []
for epoch in range(1, args.sgda_epochs + 1):

    lr = sgda_adjust_learning_rate(epoch, args, optimizer)

    print("==> SCRUB unlearning ...")

    acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
    acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
    acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, True)
    acc_rs.append(100-acc_r.item())
    acc_fs.append(100-acc_f.item())
    acc_vs.append(100-acc_v.item())

    maximize_loss = 0
    if epoch <= args.msteps:
                        train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, opt, split, quiet=False):
        maximize_loss = train_distill(epoch, forget_loader, module_list, swa_model, criterion_list, optimizer, args, "maximize")
    train_acc, train_loss = train_distill(epoch, retain_loader, module_list, swa_model, criterion_list, optimizer, args, "minimize")
    if epoch >= args.sstart:
        swa_model.update_parameters(model_s)

    
    print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
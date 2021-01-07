from history import History
import dill
import numpy as np
import os

np.set_printoptions(precision=4)
run_name = "Run5"
exp_name = "vgg2"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_hist = History(name='train_hist' + run_name)
validation_hist = History(name='validation_hist' + run_name)

train_hist = dill.load(open(ROOT_DIR + "/ckpt/" + exp_name + train_hist.name + ".pickle", "rb"))
validation_hist = dill.load(open(ROOT_DIR + "/ckpt/" + exp_name + validation_hist.name + ".pickle", "rb"))

loss, acc = np.zeros(len(train_hist.loss)), np.zeros(len(train_hist.loss))
ptp01, ptp05 = np.zeros(len(train_hist.loss)), np.zeros(len(train_hist.loss))
for i in range(len(train_hist.loss)):
    loss[i] = train_hist.loss[i].cpu()
    acc[i] = train_hist.acc[i]
    ptp01[i] = train_hist.ptp01[i]
    ptp05[i] = train_hist.ptp05[i]
epoch = train_hist.epoch
min_l = np.argmin(loss)
min_loss = loss[min_l]
min_l_dict = {'loss': loss[min_l], 'acc': acc[min_l], 'ptp01': ptp01[min_l], 'ptp05': ptp05[min_l]}
print(["{0:.3f}".format(val) for val in min_l_dict.values()])
max_a = np.argmax(acc)
max_acc = acc[max_a]
max_a_dict = {'loss': loss[max_a], 'acc': acc[max_a], 'ptp01': ptp01[max_a], 'ptp05': ptp05[max_a]}
print(["{0:.3f}".format(val) for val in max_a_dict.values()])
max_a01 = np.argmax(ptp01)
max_ptp01 = ptp01[max_a01]
max_a01_dict = {'loss': loss[max_a01], 'acc': acc[max_a01], 'ptp01': ptp01[max_a01], 'ptp05': ptp05[max_a01]}
print(["{0:.3f}".format(val) for val in max_a01_dict.values()])
max_a05 = np.argmax(ptp05)
max_ptp05 = ptp05[max_a05]
max_a05_dict = {'loss': loss[max_a05], 'acc': acc[max_a05], 'ptp01': ptp01[max_a05], 'ptp05': ptp05[max_a05]}
print(["{0:.3f}".format(val) for val in max_a05_dict.values()])

loss_v, acc_v = np.zeros(len(validation_hist.loss)), np.zeros(len(validation_hist.loss))
ptp01_v, ptp05_v = np.zeros(len(validation_hist.loss)), np.zeros(len(validation_hist.loss))
for i in range(len(validation_hist.loss)):
    loss_v[i] = validation_hist.loss[i].cpu()
    acc_v[i] = validation_hist.acc[i]
    ptp01_v[i] = validation_hist.ptp01[i]
    ptp05_v[i] = validation_hist.ptp05[i]
epoch_v = validation_hist.epoch
min_lv = np.argmin(loss_v)
min_loss_v = loss_v[min_lv]
min_lv_dict = {'loss': loss_v[min_lv], 'acc': acc_v[min_lv], 'ptp01': ptp01_v[min_lv], 'ptp05': ptp05_v[min_lv]}
print(["{0:.3f}".format(val) for val in min_lv_dict.values()])
max_av = np.argmax(acc_v)
max_acc_v = acc_v[max_av]
max_av_dict = {'loss': loss_v[max_av], 'acc': acc_v[max_av], 'ptp01': ptp01_v[max_av], 'ptp05': ptp05_v[max_av]}
print(["{0:.3f}".format(val) for val in max_av_dict.values()])
max_a01_v = np.argmax(ptp01_v)
max_ptp01_v = ptp01_v[max_a01_v]
max_a01_v_dict = {'loss': loss_v[max_a01_v], 'acc': acc_v[max_a01_v], 'ptp01': ptp01_v[max_a01_v], 'ptp05': ptp05_v[max_a01_v]}
print(["{0:.3f}".format(val) for val in max_a01_v_dict.values()])
max_a05_v = np.argmax(ptp05_v)
max_ptp05_v = ptp05_v[max_a05_v]
max_a05_v_dict = {'loss': loss_v[max_a05_v], 'acc': acc_v[max_a05_v], 'ptp01': ptp01_v[max_a05_v], 'ptp05': ptp05_v[max_a05_v]}
print(["{0:.3f}".format(val) for val in max_a05_v_dict.values()])
a = 1
# np.savez('train_hist'+run_name+'.npz', epoch=epoch, loss=loss, acc=acc, ptp01=ptp01, ptp05=ptp05)

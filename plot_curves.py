import numpy as np
import matplotlib.pyplot as plt

# f_name = '2022-04-10-14-21-30' # 50 epochs 10% data
# f_name = '2022-04-10-18-35-39' # 5 epochs 25% data
# f_name = '2022-04-10-20-16-57' # 5 epochs 50% data
f_name = '2022-04-10-22-41-57' # 5 epochs 10% data V2
f_name = '2022-04-11-10-41-47' # 10 epochs 10% data small with aux loss

dat_train = np.loadtxt(f_name + '_train.txt')
dat_test = np.loadtxt(f_name + '_test.txt')

loss_total, loss_aux, loss_model, epoch = dat_train
es = np.unique(epoch)

acc, iou = dat_test

def unfudge(xs):
    xs_total = np.array([np.diff(xs[epoch == e]) for e in es]).reshape(-1)
    xs_epoch = np.array([xs[epoch == e].mean() for e in es])

    return xs_total, xs_epoch

loss_total, loss_total_epoch = unfudge(loss_total)
loss_aux, loss_aux_epoch = unfudge(loss_aux)
loss_model, loss_model_epoch = unfudge(loss_model)

n_steps_per_epoch = (epoch == es[0]).sum()/2

# Plot
xs = np.linspace(0, es.max(), loss_total.shape[0])

fig, (ax_1, ax_2) = plt.subplots(nrows=2)
ax_1.plot(xs, loss_total, alpha=0.5, color='tab:blue', label='Total loss')
ax_1.plot(loss_total_epoch/n_steps_per_epoch, color='tab:blue')

ax_1.plot(xs, loss_model, alpha=0.5, color='tab:green', label='Model loss')
ax_1.plot(loss_model_epoch/n_steps_per_epoch, color='tab:green')

ax_1.plot(xs, loss_aux, alpha=0.5, color='tab:orange', label='Aux loss')
ax_1.plot(loss_aux_epoch/n_steps_per_epoch, color='tab:orange')

ax_2.plot(acc, label='Test accuracy')
ax_2.plot(iou, label='Test mIoU')

ax_1.grid()
ax_1.legend()
ax_2.grid()
ax_2.legend()

plt.show()
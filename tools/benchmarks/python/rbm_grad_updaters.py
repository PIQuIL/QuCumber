import numpy as np

# Updater functions based on methods explained at:
# http://cs231n.github.io/neural-networks-3/


def get_updater(method, **kwargs):
    if method == "nesterov":
        return nesterov_update, kwargs
    elif method == "momentum":
        return momentum_update, kwargs
    elif method == "adam":
        return adam_update, kwargs
    elif method == "sgd":
        return sgd_update, kwargs
    else:
        raise ValueError("`method` must be one of: 'adam', 'momentum', "
                         "'nesterov', or 'sgd'")


def sgd_update(rbm, epoch, grads, updater_data):
    lr = updater_data["learning_rate"]
    lr_current = lr(epoch)

    rbm.weights -= lr_current * grads["weights"]
    rbm.visible_bias -= lr_current * grads["visible_bias"]
    rbm.hidden_bias -= lr_current * grads["hidden_bias"]

    return {"learning_rate": lr}


def momentum_update(rbm, epoch, grads, updater_data):
    lr = updater_data["learning_rate"]
    lr_current = lr(epoch)
    mu = updater_data["momentum_param"]
    mu_current = mu(epoch)

    w_momentum = (mu_current * updater_data.get("w_momentum", 0.))
    w_momentum -= lr_current*grads["weights"]

    v_b_momentum = (mu_current * updater_data.get("v_b_momentum", 0.))
    v_b_momentum -= lr_current*grads["visible_bias"]

    h_b_momentum = (mu_current * updater_data.get("h_b_momentum", 0.))
    h_b_momentum -= lr_current*grads["hidden_bias"]

    rbm.weights += w_momentum
    rbm.visible_bias += v_b_momentum
    rbm.hidden_bias += h_b_momentum

    return {
        "learning_rate": lr,
        "momentum_param": mu,
        "w_momentum": w_momentum,
        "v_b_momentum": v_b_momentum,
        "h_b_momentum": h_b_momentum
    }


def nesterov_update(rbm, epoch, grads, updater_data):
    lr = updater_data["learning_rate"]
    lr_prev = lr(epoch - 1) if epoch > 0 else lr(0)
    mu = updater_data["momentum_param"]
    mu_prev = mu(epoch - 1) if epoch > 0 else mu(0)
    mu_current = mu(epoch)

    w_momentum = (mu_prev * updater_data.get("w_momentum", 0.))
    w_momentum -= lr_prev * grads["weights"]

    v_b_momentum = (mu_prev * updater_data.get("v_b_momentum", 0.))
    v_b_momentum -= lr_prev * grads["visible_bias"]

    h_b_momentum = (mu_prev * updater_data.get("h_b_momentum", 0.))
    h_b_momentum -= lr_prev * grads["hidden_bias"]

    # weighted sum of current and previous step's momentum values
    rbm.weights += (1 + mu_current) * w_momentum
    rbm.weights -= mu_prev * updater_data.get("w_momentum", 0.)

    rbm.visible_bias += (1 + mu_current) * v_b_momentum
    rbm.visible_bias -= mu_prev * updater_data.get("v_b_momentum", 0.)

    rbm.hidden_bias += (1 + mu_current) * h_b_momentum
    rbm.hidden_bias -= mu_prev * updater_data.get("h_b_momentum", 0.)

    return {
        "learning_rate": lr,
        "momentum_param": mu,
        "w_momentum": w_momentum,
        "v_b_momentum": v_b_momentum,
        "h_b_momentum": h_b_momentum
    }


def get_adam_momentum_values(m, v, grad, beta1, beta2, epoch):
    m = (beta1 * m) + ((1 - beta1) * grad)
    m_corrected = m / (1. - (beta1 ** (epoch + 1)))

    v = (beta2 * v) + ((1 - beta2) * (grad ** 2))
    v_corrected = v / (1. - (beta2 ** (epoch + 1)))

    return m, m_corrected, v, v_corrected


def adam_update(rbm, epoch, grads, updater_data):
    lr = updater_data["learning_rate"]
    lr_current = lr(epoch)

    beta1 = updater_data["beta1"]
    beta2 = updater_data["beta2"]
    eps = updater_data["epsilon"]

    (w_m, w_m_corrected, w_v, w_v_corrected) = \
        get_adam_momentum_values(updater_data.get("w_m", 0.),
                                 updater_data.get("w_v", 0.),
                                 grads["weights"],
                                 beta1, beta2, epoch)

    (vb_m, vb_m_corrected, vb_v, vb_v_corrected) = \
        get_adam_momentum_values(updater_data.get("vb_m", 0.),
                                 updater_data.get("vb_v", 0.),
                                 grads["visible_bias"],
                                 beta1, beta2, epoch)

    (hb_m, hb_m_corrected, hb_v, hb_v_corrected) = \
        get_adam_momentum_values(updater_data.get("hb_m", 0.),
                                 updater_data.get("hb_v", 0.),
                                 grads["hidden_bias"],
                                 beta1, beta2, epoch)

    rbm.weights -= (lr_current * w_m_corrected
                    / (np.sqrt(w_v_corrected) + eps))

    rbm.visible_bias -= (lr_current * vb_m_corrected
                         / (np.sqrt(vb_v_corrected) + eps))

    rbm.hidden_bias -= (lr_current * hb_m_corrected
                        / (np.sqrt(hb_v_corrected) + eps))

    return {
        "learning_rate": lr,
        "beta1": beta1, "beta2": beta2, "epsilon": eps,
        "w_m": w_m, "w_v": w_v,
        "vb_m": vb_m, "vb_v": vb_v,
        "hb_m": hb_m, "hb_v": hb_v
    }

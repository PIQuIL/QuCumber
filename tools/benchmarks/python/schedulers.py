
def constant(x):
    def fn(ep):
        return x
    return fn


def linear_decay(start, slope, num_epochs):
    def fn(ep):
        t = ep / float(num_epochs - 1)
        return start - (slope * t)
    return fn


def bounded_linear_decay(start, end, num_epochs):
    return linear_decay(start, start - end, num_epochs)


def linear_step_decay(start, by, every):
    def fn(ep):
        return start - (by * (ep // every))
    return fn


def exponential_decay(start, decay_rate, num_epochs):
    def fn(ep):
        t = ep / float(num_epochs - 1)
        return start * (decay_rate ** t)
    return fn


def bounded_exponential_decay(start, end, num_epochs):
    return exponential_decay(start, end/float(start), num_epochs)


def exponential_step_decay(start, by, every):
    def fn(ep):
        return start * (by ** (ep // every))
    return fn


def inverse_decay(start, k, num_epochs):
    def fn(ep):
        t = ep / float(num_epochs - 1)
        return start / (1.0 + (k*t))
    return fn


def bounded_inverse_decay(start, end, num_epochs):
    return inverse_decay(start, (start - end)/float(end), num_epochs)

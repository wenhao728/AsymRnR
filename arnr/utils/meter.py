class MetricMeter(object):
    """Meter for tracking metrics.

    Args:
        name (str): The name of the metric.
        mode (str, optional): The mode of the metric. Options ['val', 'sum', 'cnt', 'avg']. Defaults to 'avg'.
        fmt (str, optional): The format string for printing. Defaults to ':.3f'.
        alpha (float, optional): The smoothing factor for avg mode (EMA). Defaults to 0.2.
    """
    _supported_modes = ('val', 'sum', 'cnt', 'avg', 'ema')

    def __init__(self, name: str, mode: str = 'avg', fmt: str = ':.3f', alpha: float = 0.2):
        if mode not in self._supported_modes:
            raise ValueError(f'Unsupported mode={mode}, supported modes are {self._supported_modes}')
        self.name = name
        self.mode = mode
        self.fmt = fmt
        self.alpha = alpha  # smoothing factor for avg mode (EMA), e.g. 0.8 ** 10 -> 0.107
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0
        self.ema = 0

    def update(self, val: float, n: int = 1):
        reset = self.cnt == 0

        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

        if reset:
            self.ema = self.sum / self.cnt
        else:
            _alpha = 1 - (1 - self.alpha) ** n  # compatiable with n > 1
            self.ema = _alpha * self.val + (1 - _alpha) * self.ema

    def __str__(self):
        if self.mode == 'val':
            fmtstr = f'{self.name} {{val{self.fmt}}}'
        else:
            fmtstr = f'{self.name} {{val{self.fmt}}} ({{{self.mode}{self.fmt}}}){self.mode[0]}'
        return fmtstr.format(**self.__dict__)
    
    @property
    def summary(self):
        return self.__getattribute__(self.mode)
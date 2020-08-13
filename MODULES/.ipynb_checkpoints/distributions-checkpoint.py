import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from numbers import Number


class CustomLogProbTerm(TorchDistribution):
    """ This is a bogus distribution which only has the log_prob method.
        It always return custom_log_prob for any value.
        See:
        def log_prob(self, value):
            return self.custom_log_prob
    """
    arg_constraints = {'custom_log_prob': constraints.real}

    def __init__(self, custom_log_prob=None, validate_args=None):
        if custom_log_prob is None:
            raise ValueError("custom_log_prob must be specified")
        else:
            self.custom_log_prob = custom_log_prob

        if isinstance(self.custom_log_prob, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.custom_log_prob.size()
        super(CustomLogProbTerm, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CustomLogProbTerm, _instance)
        batch_shape = torch.Size(batch_shape)
        new.custom_log_prob = self.custom_log_prob.expand(batch_shape)
        super(CustomLogProbTerm, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        return self.custom_log_prob


class UnitCauchy(TorchDistribution):
    """ Cauchy distribution limited to the unit interval (0,1).

        cdf(x) = ( atan((x-loc)/scale) + N ) / ( M+N )
        pdf(x)^(-1) = (M+N) * scale * [ 1 + ((x-loc)/scale)^2 ]

        where:

        M = atan(1-loc/scale)
        N = atan(loc/scale)
    """
    arg_constraints = {'loc': constraints.unit_interval, 'scale': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.N = torch.atan(loc / scale)
        self.M_plus_N = self.N + torch.atan((1.0 - loc) / scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(UnitCauchy, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(UnitCauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.N = self.N.expand(batch_shape)
        new.M_plus_N = self.M_plus_N.expand(batch_shape)
        super(UnitCauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def cdf(self, value):
        """ Takes x in (0,1) and return something in (0,1)
            CDF(x) = 0                                   for x<=0
            CDF(x) = [ atan((x-loc)/scale) + N ] / [N+M] for 0<x<1
            CDF(x) = 1                                   for x>=1
        """
        if self._validate_args:
            self._validate_sample(value)
        tmp = torch.atan((value - self.loc) / self.scale)
        result = (tmp + self.N) / self.M_plus_N
        return result.clamp(min=0, max=1)

    def icdf(self, value):
        """ Get a number between 0 and 1 and return x in (0,1)
            Inverting CDF(x) = [ atan((x-loc)/scale) + N ] / [N+M] leads to:
            atan((x-loc)/scale) = (N+M)*CDF - N
            x = finv( f(x) )
        """

        if self._validate_args:
            self._validate_sample(value)
        tmp = value * self.M_plus_N - self.N
        return torch.clamp(self.loc + self.scale * torch.tan(tmp), min=0.0, max=1.0)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(rand)

    def log_prob(self, value):
        """ pdf(x)^(-1) = (M+N) * scale * [ 1 + ((x-loc)/scale)^2 ]  """
        if self._validate_args:
            self._validate_sample(value)
        tmp = self.scale * self.M_plus_N * (1 + ((value - self.loc) / self.scale) ** 2)
        return -torch.log(tmp)

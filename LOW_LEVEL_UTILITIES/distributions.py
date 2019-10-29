import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from numbers import Number
from math import pi as PI


class CustomLogProbTerm(TorchDistribution):
    """ This is a bogus distribution which only has the log_prob method.
        It always return custom_log_prob for any value.
        See:
        def log_prob(self, value):
            return self.custom_log_prob
    """
    arg_constraints = {'custom_log_prob': constraints.real}

    def __init__(self, custom_log_prob=None, validate_args=None):
        if custom_log_probs is None:
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
        new.custom_log_probs = self.custom_log_probs.expand(batch_shape)
        super(CustomLogProbTerm, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        return self.custom_log_prob

class NullScoreDistribution(TorchDistribution):
    
    def __init__(self, validate_args=None):
        batch_shape = torch.Size()
        super(NullScoreDistribution, self).__init__(batch_shape, validate_args=validate_args)
        
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NullScoreDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        super(NullScoreDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    def log_prob(self, x):
        return torch.zeros_like(x)
    
    
class NullScoreDistribution_Bernoulli(NullScoreDistribution):
    arg_constraints = {}
    support = constraints.integer_interval(0,1)
    has_rsample = True
    
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return (torch.rand(shape)>0.5).float()
    
class NullScoreDistribution_Unit(NullScoreDistribution):
    arg_constraints = {}
    support = constraints.unit_interval
    has_rsample = True
    
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.rand(shape)
    
    
class CustomLogProbTerm(TorchDistribution):
    """ This is a bogus distribution which only has the log_prob method.
        It always return custom_log_prob for any value:
        custom_log_prob = d.log_prob(value)
    """
    
    def __init__(self, custom_log_prob=None, validate_args=None):
        if (custom_log_prob is None):
            raise ValueError("custom_log_prob must be specified")
        self.custom_log_prob     = custom_log_prob
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
        return self.custom_log_prob.to(value.device)
    

    
class Indicator(TorchDistribution):
    """ This is a bogus distribution which only has the log_prob method.
        It has discrete support in {0,1,2,...,K-1}
        where K=log_probs.shape[-1]
        The observation are boolean (i.e. c=0,1,2,...,K-1)
        If c=0 then log_prob(c)=log_probs[0]
        If c=1 then log_prob(c)=log_probs[1]
        If c=2 then log_prob(c)=log_probs[2]
        ....
        If c=K-1 then log_prob(c)=log_probs[K-1]
        
        Typical usage:
        pyro.sample("extra_loss",Indicator(log_probs=log_probs), obs=c)
    """
    arg_constraints = {'log_probs': constraints.real}
    has_enumerate_support = False

    def __init__(self, log_probs, validate_args=None):
        if (log_probs is None):
            raise ValueError("Either `log_probs` must be specified")
        if log_probs is not None:
            if log_probs.dim() < 1:
                raise ValueError("`log_probs` parameter must be at least one-dimensional.")
            self.log_probs = log_probs
        
        self._num_events = self.log_probs.size()[-1]
        batch_shape = self.log_probs.size()[:-1] if self.log_probs.ndimension() > 1 else torch.Size()
        super(Indicator, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Indicator, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        new.log_probs = self.log_probs.expand(param_shape)
        new._num_events = self._num_events
        super(Indicator, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new


    @constraints.dependent_property
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    @property
    def param_shape(self):
        return self.log_probs.size()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.log_probs)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)


class UnitCauchy(TorchDistribution):
    """ Cauchy distribution limited to the unit interval (0,1).
        
        Rescaled Cauchy distribution:
        pdf(x) = A / { gamma*pi*[1+((x-x0)/gamma)^2] }
        cdf(x) = A* [0.5+(1/pi)*arctan((x-x0)/gamma) ] 
        
        Note that usually A = 1
        The value of A for the UnitCauchy distribution is obtained from:
        cdf(1)-cdf(0)=1 -> 
        (A/pi)*[arctan((1-x0)/gamma) + arctan(x0/gamma)] = 1 ->
        A = pi/[arctan((1-x0)/gamma) + arctan(x0/gamma)]
        
        Therefore:
        pdf_unit_cauchy(x) = A / { gamma*pi*[1+((x-x0)/gamma)^2] }
        cdf_unit_cauchy(x) = (A/pi)* [arctan((x-x0)/gamma) + arctan(x0/gamma)]
        
        the expressions for the cdf_unit_cauchy(x) can be simplified to:
        cdf_unit_cauchy(x) = (f(x)+M)/(N+M)
        where:
        f(x) = arctan((x-x0)/gamma)
        M = arctan(x0/gamma)
        N = arctan((1-x0)/gamma)
    """
    arg_constraints = {'loc': constraints.unit_interval, 'scale': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True
    
    def __init__(self, loc,scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.atan_x0   = torch.atan(loc/scale)
        self.sum       = self.atan_x0+torch.atan((1-loc)/scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(UnitCauchy, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        try:
            return self.av
        except AttributeError:
            tmp1 = ((1-self.loc)/self.scale)**2
            tmp2 = (self.loc/self.scale)**2
            tmp = (1+tmp1)/(1+tmp2)
            self.av = self.loc + torch.log(tmp)*self.scale/(2*self.sum)
            return self.av
    
    @property
    def variance(self):
        try:
            return self.var
        except AttributeError:
            self.var = self.scale/self.sum-self.scale**2-(self.mean-self.loc)**2
            return self.var
    
    @property
    def stddev(self):
        return self.variance**0.5
        
    
    @property
    def entropy(self):
        raise NotImplementedError
        
            
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(UnitCauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc   = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.atan_x0   = self.atan_x0.expand(batch_shape)
        new.sum       = self.sum.expand(batch_shape)
        super(UnitCauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

   
    def cdf(self, value):
        """ Takes x in (0,1) and return something in (0,1)
            CDF(x) = 0              for x<=0      
            CDF(x) = (f(x)+M)/(N+M) for 0<x<1 
            CDF(x) = 1              for x>=1  
        """
        if self._validate_args:
            self._validate_sample(value)
        tmp = torch.atan((value-self.loc)/self.scale)
        result = (tmp + self.atan_x0)/self.sum
        return result.clamp(min=0, max=1)
                               
    def icdf(self, value):
        """ Get a number between 0 and 1 and return x in (0,1)
            Inverting CDF(x) = (f(x)+M)/(N+M) leads to:
            f(x) = (N+M)*CDF - M
            x = finv( f(x) )
        """
        
        if self._validate_args:
            self._validate_sample(value)           
        tmp = value*self.sum - self.atan_x0
        return self.loc+self.scale*torch.tan(tmp)    

                 
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(rand)      

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)            
        tmp = self.scale*self.sum*(1+((value-self.loc)/self.scale)**2)
        return -torch.log(tmp)
    
    
class UniformWithTails(TorchDistribution):
    """ Uniform distribution between xmin and xmax with value a, 
        and exponential decay outside, i.e. a*exp[-(x-x_max)/eps]
        
        pdf(x) = a*exp[(x-xmin)/eps] for x<xmin
        pdf(x) = a                   for xmin<x<xmax
        pdf(x) = a*exp[(xmax-x)/eps] for x>xmax
             
        Normalization: -> a*(x_max-xmin)+2*a*eps=1 ==> a = 1.0/(xmax-xmin+2*eps)
        Mean: ----------> 0.5*(xmax+xmin)
        Var: -----------> Consider Exponential - troncated Exponential + Uniform 
            ------------> This gives: 0.5*a*eps*exp(delta/2*eps)*(delta^2+4delta*eps+8eps^2)+delta^3*a/12
                           
        The cumulative distribution function is: 
        CDF(x) = eps*a*exp[(x-xmin)/eps]     for x<xmin      
        CDF(x) = a*eps + a*(x-xmin)          for xmin<x<xmax 
        CDF(x) = 1 - a*eps*exp[(xmax-x)/eps] for x>xmax.
        
        Note that CDF(xmin) = a*eps
        Note that CDF(xmax) = a*eps+a*(xmax-xmin) = 1-CDF(xmin)

        where we have used the normalization 2*a*eps+a*(xmax-xmin) =1 

        Typical usage:
        d = UniformWithTails(-3.0,1.0,0.1)
        d.log_prob(torch.tensor([2.0]))
        print(d.a)
        print(d.variance)
        print(d.stddev)
        x = torch.arange(start=-50, end=50, step=0.1)  
        y = d.cdf(x)
        z = d.icdf(y)
        p = d.log_prob(x)
    """
    arg_constraints = {'low': constraints.real, 'high': constraints.real, 'eps': constraints.positive}
    support = constraints.real
    has_rsample = True
    
    def __init__(self, low, high, eps, validate_args=None):
        self.low, self.high, self.eps = broadcast_all(low, high,eps)
        #self.low  = torch.min(f,g)
        #self.high = torch.max(f,g)        
        self.a  = 1.0/(self.high-self.low+2*self.eps)
        self.cdf_low  = self.a*self.eps
        self.cdf_high = 1.0-self.cdf_low
              
        if isinstance(low, Number) and isinstance(high, Number) and isinstance(eps, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super(UniformWithTails, self).__init__(batch_shape, validate_args=validate_args)

        if self._validate_args and not torch.lt(self.low, self.high).all():
            raise ValueError("UniformWithTails is not defined when low>= high")
        if self._validate_args and not (self.eps > 0).all():
            raise ValueError("UniformWithTails is not defined when eps <= 0")

    @property
    def mean(self):
        return (self.high + self.low) / 2

    @property
    def stddev(self):
        try:
            return self.var**0.5
        except AttributeError:
            delta = self.high-self.low
            tmp1 = delta.pow(3)*self.a/12 
            tmp2 = torch.exp(-0.5*delta/self.eps)
            tmp3 = (0.5*self.a*self.eps*(delta.pow(2)+4*self.eps*(delta+2*self.eps)))
            self.var = tmp1 + tmp2*tmp3
            return self.var**0.5
   
    @property
    def variance(self):
        try:
            return self.var
        except AttributeError:
            delta = self.high-self.low
            tmp1 = delta.pow(3)*self.a/12 
            tmp2 = torch.exp(-0.5*delta/self.eps)
            tmp3 = (0.5*self.a*self.eps*(delta.pow(2)+4*self.eps*(delta+2*self.eps)))
            self.var = tmp1 + tmp2*tmp3
            return self.var
    
    @property
    def entropy(self):
        raise NotImplementedError
        
            
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(UniformWithTails, _instance)
        batch_shape = torch.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        new.eps = self.eps.expand(batch_shape)
        new.a  = 1.0/(new.high-new.low+2*new.eps)
        new.cdf_low  = new.a*new.eps
        new.cdf_high = 1.0-new.cdf_low       
        super(UniformWithTails, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

   
    def cdf(self, value):
        """ Takes x in (-Infinity,Infinity) and return something in (0,1)
            CDF(x) = eps*a*exp[(x-xmin)/eps]     for x<xmin      
            CDF(x) = a*eps + a*(x-xmin)           for xmin<x<xmax 
            CDF(x) = 1 - a*eps*exp[(xmax-x)/eps] for x>xmax. 
            Note that CDF(xmin) = a*eps, CDF(xmax) = 1 - a*eps
        """
        if self._validate_args:
            self._validate_sample(value)
              
        result1 = self.cdf_low * torch.exp((value-self.low)/self.eps)
        result2 = self.cdf_low + self.a*(value-self.low)
        result3 = 1.0-self.cdf_low*torch.exp((self.high-value)/self.eps)
        
        mask_low  = (value < self.low).float()
        mask_high = (value > self.high).float()
        mask_middle = (1-mask_low)*(1-mask_high)
        
        # Unfortunately: (mask=0)*(result = Inf)=Nan
        # Therefore I need to remove the Inf manually
        
        result1[result1 == float("-Inf")] = 0
        result2[result2 == float("-Inf")] = 0
        result3[result3 == float("-Inf")] = 0
        
        result1[result1 == float("+Inf")] = 0
        result2[result2 == float("+Inf")] = 0
        result3[result3 == float("+Inf")] = 0
        
        result = mask_low*result1 + mask_middle*result2 + mask_high*result3 
        return result.clamp(min=0, max=1)
                               
    def icdf(self, value):
        """ Get a number between 0 and 1 and return x in (-Infinity,Infinity)
            CDF(x) = eps*a*exp[(x-xmin)/eps]     for x<xmin      
            CDF(x) = a*eps + a*(x-xmin)           for xmin<x<xmax 
            CDF(x) = 1 - a*eps*exp[(xmax-x)/eps] for x>xmax. 
            Note that CDF(xmin) = a*eps, CDF(xmax) = 1 - a*eps
        """
        
        if self._validate_args:
            self._validate_sample(value)
            
        result1 = self.low+self.eps*torch.log(value/self.cdf_low) 
        result2 = self.low+(value-self.cdf_low)/self.a
        result3 = self.high-self.eps*torch.log((1.0-value)/self.cdf_low)
        
        mask_low  = (value < self.cdf_low).float()
        mask_high = (value > self.cdf_high).float()
        mask_middle = (1-mask_low)*(1-mask_high)
              
        #print("result",result1,result2,result3)
        #print("mask",mask_low,mask_middle,mask_high)
            
        result = mask_low*result1 + mask_middle*result2 + mask_high*result3 
        return result
                 
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.icdf(rand)      


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        #result1 = torch.log(self.a)-(self.low-value)/self.eps
        #result2 = torch.log(self.a)
        #result3 = torch.log(self.a)-(value-self.high)/self.eps
        #result = mask_low*result1 + mask_middle*result2 + mask_high*result3 
        
        # Since torch.log(self.a) appears everywhere I can simplify the expression
        result1 = -(self.low-value)/self.eps
        result3 = -(value-self.high)/self.eps        
        mask_low  = (value < self.low).float()
        mask_high = (value > self.high).float()              
        result = torch.log(self.a) + mask_low*result1 + mask_high*result3 
        
        return result
    

###class Exp_shift_rate(TorchDistribution):
###    """ This is a bogus distribution which only has the log_prob method.
###        
###        p(x|shift,rate) = 0.5*rate*exp(-rate*||x-shift||)
###        
###        log p(x|shift,rate) = log(0.5*rate)-rate*||x-shift||
###        
###        CDF(x) = 0.5*exp(-rate*||shift-x||)        for x<shift      
###        CDF(x) = 1.0 - 0.5*exp(-rate*||shift-x||)  for x>shift 
###    """
###
###    arg_constraints = {'shift': constraints.real, 'rate': constraints.positive, 'scale':constraints.positive}
###    support = constraints.real
###    has_enumerate_support = True
###
###
###    def __init__(self, shift=None, rate=None, scale=None, validate_args=None):
###        
###        if(shift is None):
###            raise ValueError("Shift must be specified")
###        else:
###            if((rate is None) == (scale is None)):
###                raise ValueError("Only (and only one) between rate and scale must be specified")
###            elif(rate is None):
###                rate = 1.0/scale
###                
###        self.shift, self.rate = broadcast_all(shift,rate)
###        
###        if isinstance(shift, Number) and isinstance(rate, Number):
###            batch_shape = torch.Size()
###        else:
###            batch_shape = self.shift.size()
###        super(Exp_shift_rate, self).__init__(batch_shape, validate_args=validate_args)
###        
###    @property
###    def mean(self):
###        return self.shift
###
###    @property
###    def stddev(self):
###        try:
###            return self.var**0.5
###        except AttributeError:
###            self.var = 2.0/self.rate**2
###            return self.var**0.5
###   
###    @property
###    def variance(self):
###        try:
###            return self.var
###        except AttributeError:
###            self.var = 2.0/self.rate**2
###            return self.var
###    
###    @property
###    def entropy(self):
###        raise NotImplementedError
###        
###    def expand(self, batch_shape, _instance=None):
###        new = self._get_checked_instance(Exp_shift_rate, _instance)
###        batch_shape = torch.Size(batch_shape)
###        new.shift = self.shift.expand(batch_shape)
###        new.rate  = self.rate.expand(batch_shape)
###        
###        super(Exp_shift_rate, new).__init__(batch_shape, validate_args=False)
###        new._validate_args = self._validate_args
###        return new
###    
###    def cdf(self, value):
###        """ Takes x in (-Infinity,Infinity) and return something in (0,1)
###            CDF(x) = 0.5*exp(-rate*||shift-x||)        for x<shift      
###            CDF(x) = 1.0 - 0.5*exp(-rate*||shift-x||)  for x>shift 
###        """
###        if self._validate_args:
###            self._validate_sample(value)
###             
###        tmp  = 0.5*torch.exp(-self.rate*torch.abs(self.shift-value))
###        return torch.where(value<self.shift,tmp,1.0-tmp) 
###    
###    
###    def icdf(self, value):
###        """ Get a number between 0 and 1 and return x in (-Infinity,Infinity) """
###        
###        if self._validate_args:
###            self._validate_sample(value)
###        
###        c = torch.min(value,1.0-value)
###        tmp = torch.log(2*c)/self.rate
###        return torch.where(value<0.5,self.shift+tmp,self.shift-tmp)
###    
###    def rsample(self, sample_shape=torch.Size()):
###        shape = self._extended_shape(sample_shape)
###        rand = torch.rand(shape, dtype=self.shift.dtype, device=self.shift.device)
###        return self.icdf(rand)   
###      
###    def log_prob(self, value):
###        one = value.new_ones(1)
###        return torch.log(0.5*self.rate*one)-self.rate*torch.abs(value-self.shift)
###    
###    
###    
###class SlopeWithTails(TorchDistribution):
###""" Exponential tails + linear distribution in the middle.
###    The parametrization is:
###    1. xmin,xmax = interval over which the distribution is linear
###    2. eps = scale of the exponential tails: exp[-(x-xmax)/exp]
###    3. ratio = value of p(xmax)/p(xmin)
###"""     
   
    
    

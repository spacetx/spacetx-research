import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from numbers import Number
from math import pi as PI

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
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return (torch.rand(shape)>0.5).float()

class NullScoreDistribution_Bernoulli(NullScoreDistribution):
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return (torch.rand(shape)>0.5).float()
            
class NullScoreDistribution_Unit(NullScoreDistribution):
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

    
class IndicatorNew(TorchDistribution):
    """ This is a bogus distribution which only has the log_prob method.
        It has discrete support in {0,1,2,...,K-1}
        where K=logits.shape[-1]
        The observation are boolean (i.e. c=0,1,2,...,K-1)
        If c=0 then log_prob(c)=logits[0]
        If c=1 then log_prob(c)=logits[1]
        If c=2 then log_prob(c)=logits[2]
        ....
        If c=K-1 then log_prob(c)=logits[K-1]
        
        Typical usage:
        pyro.sample("extra_loss",Indicator(logits=logits), obs=c)
    """

    arg_constraints = {'logits': constraints.real}
    support = constraints.integer_interval
    has_enumerate_support = True


    def __init__(self, logits=None, validate_args=None):
        
        if (logits is None):
            raise ValueError("logits must be specified")
            
        self.logits     = logits 
        self.num_events = self.logits.size()[-1]
        batch_shape = self.logits.size()[:-1] if self.logits.ndimension() > 1 else torch.Size()
        super(IndicatorNew, self).__init__(batch_shape, validate_args=validate_args)
        

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def stddev(self):
        raise NotImplementedError
   
    @property
    def variance(self):
        raise NotImplementedError
    
    @property
    def entropy(self):
        raise NotImplementedError
            
    def cdf(self, value):
        raise NotImplementedError
                               
    def icdf(self, value):
        raise NotImplementedError
                 
    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(IndicatorNew, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self.num_events,))

        new.logits = self.logits.expand(param_shape)
        new.num_events = new.logits.size()[-1]
        super(IndicatorNew, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        #if value = 0 return self.logits[0]
        #if value = 1 return self.logits[1]
        #if value = 2 return self.logits[2]

        if self._validate_args:
            self._validate_sample(value)
           
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)


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

    
    
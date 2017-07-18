import numpy as np
from scipy.stats import  multivariate_normal,truncnorm
from itertools import izip
import warnings


class catsampling:
    def __init__(self,categories,discprobs,gamma):
        self.categories = categories
        self.q = len(categories)
        if discprobs is None:
            self.discprobs = [np.array([1.0/len(categories[i])]*len(categories[i]))
                                       for i in xrange(self.q)]
        else:
            self.discprobs = [np.array(x) for x in discprobs]
        self.gamma = gamma
    def rvs(self,N):
        return sample_categorical(self.discprobs,N,self.categories)
    
    def update(self,X,gamma = None):
        if gamma is None:
            gamma = self.gamma
        for i,cat_vals,cat in izip(xrange(self.q),izip(*X),self.categories):
            count = np.array(tuple(cat_vals.count(val) for val in cat),dtype=np.float)
            self.discprobs[i] = count/count.sum()*gamma+self.discprobs[i]*(1.0-gamma)
            self.max_prob_gap = max((1.0-max(x)) for x in self.discprobs)
        
        
        
class contsampling:
    def __init__(self,means,sigmas,a,b,alpha=0.9,beta=0.9):
        self.trunc = False
        self.means = means
        self.alpha = alpha
        self.beta = beta
        if a == None and b == None:
            #Use a multivariate normal
            self.Xcov = np.diag(sigmas)**2
            self.rv = multivariate_normal(means, self.Xcov)
        else:
            #Use several truncated normals
            self.rv = truncnormmulti(means,sigmas,a,b)
            self.trunc = True
            self.a = a
            self.b = b
            
    def rvs(self,N):
        return self.rv.rvs(N)
    
    def update(self,X):
        self.means = X.mean(axis=0)*self.alpha + self.means*(1.0-self.alpha)
        if self.trunc:
            #update standard deviations
            self.sigmas = X.std(axis=0)*self.beta + self.sigmas*(1.0-self.beta)
            self.rv = truncnormmulti(self.means,self.sigmas,self.a,self.b)
            rv.maxsd = np.max(sigmas)
        else: 
            #update covariance matrix
            self.Xcov = np.cov(X.T)*self.beta+self.Xcov*(1.0-self.beta)
            self.rv = multivariate_normal(self.means, self.Xcov)
            self.maxsd = np.sqrt(np.max(self.Xcov))


class truncnormmulti:
    def __init__(self,mus,sigmas,lowers,uppers):
        self.rand_vars = [truncnorm((lower - mu) / sigma, 
                                    (upper - mu) / sigma, loc=mu, 
                                    scale=sigma) 
                                    for mu,sigma,lower,upper in izip(mus,sigmas,lowers,uppers)]
    def rvs(self,n):
        X = np.vstack((rv.rvs(n) for rv in self.rand_vars))
        return X.T

class categorical_handler:
    def __init__(self,categories):
        if isinstance(categories,dict):
            self.cat_dict = True
            self.cats = []
            idx = 0 
            self.idx_to_key = {}
            for key,values in categories.iteritems():
                self.cats.append(values)
                self.idx_to_key[idx]=key
                idx+=1
                            
        elif isinstance(categories,list):
            self.cat_dict = False
            self.cats = categories
        else:
            raise ValueError("categorical_handler expects a list or dictionary")    
            
    def get_categories(self):
        return self.cats
    
    def handle(self,Xd):
        if self.cat_dict:
            return {self.idx_to_key[i]:val for i,val in enumerate(Xd)}
        else:
            return Xd
    
    
def sample_categorical(discrete_probs,N,categories):
    X = [x for x in gen_cat_sample(discrete_probs,N,categories)]
    return zip(*X)
    
def gen_cat_sample(discrete_probs,N,categories):
    for probs,cats in izip(discrete_probs,categories): 
        cutoffs = np.array(probs).cumsum()
        x = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1],N))
        yield [cats[i] for i in x]

def cemethod(objf,fargs= None,fkwargs= None,maximize=False,continuous = None,discrete = None,
             N = 100,rho = 0.1,iterThr = 1e4,noImproveThr = 5,verbose = True,savestates = False):
    # Argument continuous should be a dictionary specifying the sampling distribution 
    # specifing the continuous optimization variables.
    
    if not callable(objf):
        raise ValueError("Objective function is not callable")
    
    # check argument continuous
    continuous_default ={
        'smooth_mean' : 0.9, # smoohing parameter for mean
        'smooth_sd' : 0.9, # smoohing parameter for the standard deviation
        'sd_threshold' : 0.001, # threshold for largest standard deviation
        'lower' : None,
        'upper' : None,
        'mu' : None,
        'sigma' : None
    }

    if not continuous is None:
        if isinstance(continuous,dict):
            for key in continuous.keys():
                if not key in continuous_default:
                    raise KeyError('Unknown key in argument continuous: %s'%key)
            continuous_default.update(continuous)
        else:
            raise TypeError("Argument continuous should be a dictionary")

    # unpack continuous
    alpha = continuous_default['smooth_mean']
    beta = continuous_default['smooth_sd']
    eps = continuous_default['sd_threshold']
    a = continuous_default['lower']
    b = continuous_default['upper']
    means = continuous_default['mu']
    sigmas = continuous_default['sigma']
    
    # check continuous settings
    if (not means is None) or (not sigmas is None):
        if (means is None) or (sigmas is None):  
            raise ValueError("If mu is specified " 
                             "sigma needs to be specified as well and vice vesa")
        
        if isinstance(means, list):
            means = np.array(means)
        if not isinstance(means, np.ndarray):
            raise TypeError("Argument mu should be a list or array")
        p1 = len(means)
        
        if isinstance(sigmas, list):
            sigmas = np.array(sigmas)
        if not isinstance(sigmas, np.ndarray):
            raise TypeError("Argument sigma should be a list or array")
        p2 = len(sigmas)
        
        if p2!=p1:
            raise ValueError("mu and sigma need to be of the same length")
        else:
            p = p1
    
    # check continuous constraints
    truncate = False
    if (not a is None) or (not b is None):
        truncate = True
        if not a is None:
            if isinstance(a, list):
                a = np.array(a)
            if not isinstance(a, np.ndarray):
                raise TypeError("Argument lower should be a list or array")
            if len(a)!= p:
                raise ValueError("lower needs to be None or of the same length as mu and sigma")
        else:
            a = np.full((p,),-np.infty)
            
        if not b is None:
            if isinstance(b, list):
                b = np.array(b)
            if not isinstance(b, np.ndarray):
                raise TypeError("Argument upper should be a list or array")
            if len(b)!= p:
                raise ValueError("upper needs to be None or of the same length as mu and sigma")
        else:
            b = np.full((p,),np.infty)
            
    # check argument discrete
    discrete_default = {
        'categories' : None,
        'probs' : None,
        'gamma' : 0.9, # smoohing parameter for categorical sampling probabilities
        'prob_threshold' : 0.001 # threshold for largest gap to probability of 1.0
    }
            
    if not discrete is None:
        if isinstance(discrete,dict):
            for key in discrete.keys():
                if not key in discrete_default:
                    raise KeyError('Unknown key in argument discrete: %s'%key)
            discrete_default.update(discrete)
        else:
            raise TypeError("Argument discrete should be a dictionary")
    
    gamma = discrete_default['gamma']
    eta = discrete_default['prob_threshold']
    discprobs = discrete_default['probs']
    categories = discrete_default['categories']

    handle_cat = None
    
    if not categories is None:
        q = len(categories)
        if not isinstance(categories,list):
            if not isinstance(categories,dict):
                raise TypeError("Argument categories should be a list or dictionary")
            
        if not discprobs is None:
            if not isinstance(discprobs, list):
                raise TypeError("Argument probs should be a list ")
            if len(discprobs)!= q:
                raise ValueError("probs needs to be None or of the same length "
                             "as the number of discrete variables in categories")
            if isinstance(categories,list):
                for i,probrow,cat in izip(xrange(q),discprobs,categories):
                    if len(probrow)!=(categories):
                        raise ValueError("number of probabilities at index %d"
                                         "do not match numbe of categories")
                        
            else:
                for i,probrow,cat in izip(xrange(q),discprobs,categories):
                    if len(probrow)!=(categories):
                        raise ValueError("number of probabilities at index %d"
                                         "do not match numbe of categories")

        # wrap categories such that list or dict can be used
        handle_cat = categorical_handler(categories)
        categories = handle_cat.get_categories() # returns a list
                    
    #initialize variables
    if fargs is None:
        fargs = ()
    if fkwargs is None:
        fkwargs = {}
    
    n_elite = int(np.round(N*rho))
    if n_elite <= 1:
        n_elite=2
        warnings.warn("rho and/or N too small")
    
    Xc = None
    Xd = None
        
    mc = 1.0
    if maximize:
        mc = -1.0

    # Wrap objective function
    if p>0 and q > 0:
        def fwrap(xc,xd):
            return mc*objf(xc,handle_cat.handle(xd),*fargs,**fkwargs)
    elif p>0:
        def fwrap(xc,xd):
            return mc*objf(xc,*fargs,**fkwargs)
    elif q>0:
        def fwrap(xc,xd):
            return mc*objf(handle_cat.handle(xd),*fargs,**fkwargs)
    else:
        raise RuntimeError("At least one argument of continuous or discrete has to be specified")

    # initialize random variables and generate initial sample
    if p >0:
        rv = contsampling(means,sigmas,a,b)
        Xc = rv.rvs(N)
        if p ==1:
            Xc = Xc.reshape(N,p)

    if q > 0:
        #discrete sample
        d_rv = catsampling(categories,discprobs,gamma)
        Xd = d_rv.rvs(N)


    # Evaluate initial sample
    Y = np.array(tuple(fwrap(xc,xd) for xc,xd in izip(Xc,Xd)))

    # Figure out the elite
    elite = np.argpartition(Y, n_elite)[:n_elite] # argpartition n_elite smallest

    elite_vals = Y[elite]
    # now estimate the sampling distributions
    if p>0:
        rv.update(Xc[elite])
    if q>0:
        Xde = [Xd[i] for i in elite]
        d_rv.update(Xde)



    iteration = 0
    elmin = elite[np.argmin(elite_vals)]

    c_opt = Xc[elmin] #best Xc
    d_opt = Xd[elmin]#best Xd
    optimum = Y[elmin]
    gammat = max(elite_vals)
    ceprocess = []
    diffopt = np.infty
    states = []
    probst = list()
    max_prob_gap = d_rv.max_prob_gap
    # CEmethod loop
    while (iteration < iterThr and diffopt!=0 and
           (( p>0 and rv.maxsd > eps) or
            (q>0 and max_prob_gap > eta))):
        if savestates or verbose:
            tmpdct = {'iteration':iteration,'optimum':optimum*mc,'gammat':gammat*mc}
            if p>0:
                tmpdct.update({'max sd':rv.maxsd})
            if q>0:
                tmpdct.update({'max prob gap':max_prob_gap})
            if verbose:
                print '%s:\t'%tmpdct['iteration'],
                print ', '.join([('%s: %.5f'%(key,val)) for key,val in tmpdct.iteritems() if key!='iteration'])
                print
            if savestates:
                states.append(tmpdct)

        # generate samples
        if p >0:
            #cont sample
            Xc = rv.rvs(N)
            if p ==1:
                Xc = Xc.reshape(N,p)

        if q > 0:
            #discrete sample
            Xd = d_rv.rvs(N)

        # evaluate objective
        Y = np.array(tuple(fwrap(xc,xd) for xc,xd in izip(Xc,Xd)))

        # Figure out the elite
        # no need to sort completely
        elite = np.argpartition(Y, n_elite)[:n_elite] # argpartition n_elite smallest
        elite_vals = Y[elite]
        elmin = elite[np.argmin(elite_vals)]

        # update the sampling distributions
        if p>0:
            rv.update(Xc[elite])
        if q>0:
            Xde = [Xd[i] for i in elite]
            d_rv.update(Xde)    
            max_prob_gap = d_rv.max_prob_gap
        # check for optimum
        if Y[elmin] < optimum:
            if p>0: 
                c_opt = Xc[elmin]
            if q>0: 
                d_opt = Xd[elmin]
            optimum = Y[elmin]
            gmax = max(elite_vals)
            if gmax<gammat:
                gammat = gmax

        ceprocess.append(optimum)
        if iteration > noImproveThr:
            diffopt = np.sum(np.abs(np.array(ceprocess[iteration-noImproveThr:])-optimum))
        iteration+=1
        
    if savestates or verbose:
        tmpdct = {'iteration':iteration,'optimum':optimum*mc,'gammat':gammat*mc}
        if p>0:
            tmpdct.update({'max sd':rv.maxsd})
        if q>0:
            tmpdct.update({'max prob gap':max_prob_gap})
        if verbose:
            print '%s:\t'%tmpdct['iteration'],
            print ', '.join([('%s: %.5f'%(key,val)) for key,val in tmpdct.iteritems() if key!='iteration'])
            print
        if savestates:
            states.append(tmpdct)

    # create output, termination criterion, elite, etc. 

    # check termination criteria
    if iteration==iterThr:
        convergence="Not converged" 
    elif diffopt==0: 
        convergence= "Optimum did not change for %d iterations"%noImproveThr
    else:
        convergence="Variance converged"


    out = {'optimizer':{'continuous':c_opt,'discrete':d_opt},
     'optimum':optimum*mc,
     'termination':{'iteration':iteration,
                    'function evaluations':iteration*N,
                    'convergence':convergence
                   }
    }
    if savestates:
        out.update({'states':states})
    
    return out

if __name__ == "__main__":
    def myfun(x,cats):
        a = 0
        if 1 == cats[0]:
            a += 2.0
        if 'b' == cats[1]:
            a += 2.0

        return sum(np.square(x))+a

    _ = cemethod(myfun,maximize=False,continuous = {'mu':np.ones(1)*4,'sigma':np.ones(1),'smooth_mean' : 1.0,'smooth_sd' : 1.0,'sd_threshold' : 0.1},
         discrete = {'categories':[[1,5,2],['a','b','c']]},rho=0.1)

"""Linear ballistic accumulator for use with pymc3."""

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import math
import scipy.stats as st
from nireact import model


def sample_finish_time(A, b, v, s, tau, size):
    """Sample finish time for a set of accumulators."""
    # select starting point
    k = st.uniform.rvs(loc=0, scale=A, size=size)

    t = np.zeros((len(v), size))
    for i, vi in enumerate(v):
        # sample drift rate, calculate time to threshold
        d = st.norm.rvs(loc=vi, scale=s, size=size)
        ti = tau + ((b - k) / d)

        # time is invalid if drift rate is negative
        ti[d < 0] = np.nan
        t[i, :] = ti
    return t


def sample_response(A, b, v, s, tau, size):
    """Sample response from a set of accumulators."""

    # get finish time for each accumulator
    t = sample_finish_time(A, b, v, s, tau, size)

    # determine winner on each valid trial
    valid = np.any(np.logical_not(np.isnan(t)), 0)
    t_valid = t[:, valid]
    t_winner = np.nanmin(t_valid, 0)
    i_winner = np.nanargmin(t_valid, 0)

    # initialize full matrix
    response = np.empty(size)
    response.fill(np.nan)
    rt = np.empty(size)
    rt.fill(np.nan)

    # fill in valid trials
    rt[valid] = t_winner
    response[valid] = i_winner
    return rt, response


def normpdf(x):
    return (1 / pm.math.sqrt(2 * math.pi)) * pm.math.exp(-(x ** 2) / 2)


def normcdf(x):
    return (1 / 2) * (1 + pm.math.erf(x / pm.math.sqrt(2)))


def tpdf(t, A, b, v, sv):
    """Probability distribution function over time."""
    g = (b - A - t * v) / (t * sv)
    h = (b - t * v) / (t * sv)
    f = (-v * normcdf(g) + sv * normpdf(g) +
         v * normcdf(h) - sv * normpdf(h)) / A
    return f


def tcdf(t, A, b, v, s):
    """Cumulative distribution function over time."""
    e1 = ((b - A - t * v) / A) * normcdf((b - A - t * v) / (t * s))
    e2 = ((b - t * v) / A) * normcdf((b - t * v) / (t * s))
    e3 = ((t * s) / A) * normpdf((b - A - t * v) / (t * s))
    e4 = ((t * s) / A) * normpdf((b - t * v) / (t * s))
    F = 1 + e1 - e2 + e3 - e4
    return F


def tpdfi(t, i, A, b, v1, v2, s):
    """PDF for accumulator i."""
    # probability that all accumulators are negative
    all_neg = normcdf(-v1 / s) * normcdf(-v2 / s)
    pdf = tt.switch(tt.eq(i, 1),
                    ((1 - tcdf(t, A, b, v2, s)) *
                     tpdf(t, A, b, v1, s)) / (1 - all_neg),
                    ((1 - tcdf(t, A, b, v1, s)) *
                     tpdf(t, A, b, v2, s)) / (1 - all_neg))
    pdf_cond = tt.switch(tt.gt(t, 0), pdf, 0)
    return pdf_cond


def tpdfvar(t, i, n, A, b, v1, v2, r, s):
    """PDF for variable speed model."""

    # drift rate depends on number of retrievals
    v1a = tt.switch(tt.eq(n, 1), v1, v1 * r)
    v2a = tt.switch(tt.eq(n, 1), v2, v2 * r)

    # after drift rate adjustment, same PDF as basic model
    pdf_cond = tpdfi(t, i, A, b, v1a, v2a, s)
    return pdf_cond


def tpdfsep(t, i, n, A, b, v1, v2, v3, v4, s):
    """PDF for separate process model."""

    vc = tt.switch(tt.eq(n, 1), v1, v3)
    vi = tt.switch(tt.eq(n, 1), v2, v4)
    pdf_cond = tpdfi(t, i, A, b, vc, vi, s)
    return pdf_cond


def tpdfnav(t, i, n, A, b, v1, v2, r, v3, v4, s):
    """PDF for navigation model."""

    # drift rate depends on number of retrievals
    v1a = tt.switch(tt.eq(n, 1), v1, v1 * r)
    v2a = tt.switch(tt.eq(n, 1), v2, v2 * r)

    # probability that all accumulators are negative
    all_neg = (normcdf(-v1a / s) * normcdf(-v2a / s) *
               normcdf(-v3 / s) * normcdf(-v4 / s))

    # PDF for each accumulator
    p1 = tpdf(t, A, b, v1a, s)
    p2 = tpdf(t, A, b, v2a, s)
    p3 = tpdf(t, A, b, v3, s)
    p4 = tpdf(t, A, b, v4, s)

    # probability of having not hit threshold by now
    n1 = 1 - tcdf(t, A, b, v1a, s)
    n2 = 1 - tcdf(t, A, b, v2a, s)
    n3 = 1 - tcdf(t, A, b, v3, s)
    n4 = 1 - tcdf(t, A, b, v4, s)

    # conditional probability of each accumulator hitting threshold now
    c1 = p1 * n2 * n3 * n4
    c2 = p2 * n1 * n3 * n4
    c3 = p3 * n1 * n2 * n4
    c4 = p4 * n1 * n2 * n3

    # calculate probability of this response and rt,
    # conditional on a valid response
    pdf = tt.switch(tt.eq(i, 1),
                    (c1 + c3) / (1 - all_neg),
                    (c2 + c4) / (1 - all_neg))
    pdf_cond = tt.switch(tt.gt(t, 0), pdf, 0)
    return pdf_cond


def tpdfi_func(scalar=False):
    theano.config.compute_test_value = 'ignore'
    if scalar:
        t = tt.dscalar('t')
        i = tt.iscalar('i')
        n = tt.iscalar('n')
    else:
        t = tt.dvector('t')
        i = tt.lvector('i')
        n = tt.lvector('n')
    A = tt.dscalar('A')
    b = tt.dscalar('b')
    v1 = tt.dscalar('v1')
    v2 = tt.dscalar('v2')
    s = tt.dscalar('s')
    tau = tt.dscalar('tau')
    t0 = t - tau
    pdf = tpdfi(t0, i, A, b, v1, v2, s)
    return theano.function([t, i, n, A, b, v1, v2, s, tau], pdf,
                           on_unused_input='ignore')


def tpdfvar_func(scalar=False):
    """Generate PDF function for simple LBA."""

    theano.config.compute_test_value = 'ignore'
    if scalar:
        t = tt.dscalar('t')
        i = tt.iscalar('i')
        n = tt.iscalar('n')
    else:
        t = tt.dvector('t')
        i = tt.lvector('i')
        n = tt.lvector('n')
    A = tt.dscalar('A')
    b = tt.dscalar('b')
    v1 = tt.dscalar('v1')
    v2 = tt.dscalar('v2')
    r = tt.dscalar('r')
    s = tt.dscalar('s')
    tau = tt.dscalar('tau')
    t0 = t - tau
    pdf = tpdfvar(t0, i, n, A, b, v1, v2, r, s)
    return theano.function([t, i, n, A, b, v1, v2, r, s, tau], pdf)


def tpdfsep_func(scalar=False):
    """Generate PDF function for separate process LBA."""

    theano.config.compute_test_value = 'ignore'
    if scalar:
        t = tt.dscalar('t')
        i = tt.iscalar('i')
        n = tt.iscalar('n')
    else:
        t = tt.dvector('t')
        i = tt.lvector('i')
        n = tt.lvector('n')
    A = tt.dscalar('A')
    b = tt.dscalar('b')
    v1 = tt.dscalar('v1')
    v2 = tt.dscalar('v2')
    v3 = tt.dscalar('v3')
    v4 = tt.dscalar('v4')
    s = tt.dscalar('s')
    tau = tt.dscalar('tau')
    t0 = t - tau
    pdf = tpdfsep(t0, i, n, A, b, v1, v2, v3, v4, s)
    return theano.function([t, i, n, A, b, v1, v2, v3, v4, s, tau], pdf)


def tpdfnav_func(scalar=False):
    """Generate PDF function for navigation model."""
    theano.config.compute_test_value = 'ignore'
    if scalar:
        t = tt.dscalar('t')
        i = tt.iscalar('i')
        n = tt.iscalar('n')
    else:
        t = tt.dvector('t')
        i = tt.lvector('i')
        n = tt.lvector('n')
    A = tt.dscalar('A')
    b = tt.dscalar('b')
    v1 = tt.dscalar('v1')
    v2 = tt.dscalar('v2')
    r = tt.dscalar('r')
    v3 = tt.dscalar('v3')
    v4 = tt.dscalar('v4')
    s = tt.dscalar('s')
    tau = tt.dscalar('tau')
    pdf = tpdfnav(t - tau, i, n, A, b, v1, v2, r, v3, v4, s)
    return theano.function([t, i, n, A, b, v1, v2, r, v3, v4, s, tau], pdf)


def tpdfi_rvs(A, b, v1, v2, s, tau, size=1):
    """Random sampler for two accumulators."""

    # sample response from accumulators
    rt, resp = sample_response(A, b, [v1, v2], s, tau, size)

    # if accumulator 1 finished first, trial was correct
    correct = np.zeros(size)
    correct[resp == 0] = 1
    return rt, correct


def tpdfvar_rvs(n, A, b, v1, v2, r, s, tau, size=1):
    """Random generator for variable-drift model."""

    if n == 2:
        v1 *= r
        v2 *= r
    rt, correct = tpdfi_rvs(A, b, v1, v2, s, tau, size=size)
    return rt, correct


def tpdfsep_rvs(n, A, b, v1, v2, v3, v4, s, tau, size=1):
    """Random generator for separate process model."""

    vc = v1 if n == 1 else v3
    vi = v2 if n == 1 else v4
    rt, correct = tpdfi_rvs(A, b, vc, vi, s, tau, size=size)
    return rt, correct


def tpdfnav_rvs(n, A, b, v1, v2, r, v3, v4, s, tau, size=1):
    """Random generator for navigation model."""

    # account for number of retrievals required for test type
    if n == 2:
        v1 = v1 * r
        v2 = v2 * r

    # finish times of accumulators
    rt, resp = sample_response(A, b, [v1, v2, v3, v4], s, tau, size)

    # accumulators 1 and 3 are both for the correct response
    correct = np.zeros(size)
    correct[np.isin(resp, [0, 2])] = 1
    return rt, correct


def annotate_graph(graph, size=1):
    """Add annotations to graph to support plotting."""
    for var in graph.unobserved_RVs:
        var.id = var.name

    for var in graph.observed_RVs:
        var.id = var.name
        var.tag = Scratch(test_value=np.zeros(size))


class Scratch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LBA(model.Model):
    """Linear Ballistic Accumulator model."""

    def tensor_pdf(self, rt, response, test, param):
        tau = param['tau']
        sub_param = param.copy()
        del sub_param['tau']
        return tpdfi(rt - tau, response, **sub_param)

    def function_pdf(self):
        return tpdfi_func()

    def rvs_test(self, test, param, size):
        rt, response = tpdfi_rvs(**param, size=size)
        return rt, response

    def init_graph(self, rt, response, test):
        data = {'rt': rt, 'response': response, 'test': test}
        with pm.Model() as graph:
            s = 1
            tau = .2
            b = 12
            A = pm.Uniform('A', lower=0, upper=b)
            v1 = pm.Uniform('v1', lower=0, upper=10)
            v2 = pm.Uniform('v2', lower=0, upper=10)
            param = {'A': A, 'b': b, 'v1': v1, 'v2': v2, 's': s, 'tau': tau}
            logp = self.tensor_logp(param)
            t_resp = pm.DensityDist('t_resp', logp, observed=data)
        return graph


class LBAVar(model.Model):
    """Variable-drift LBA model."""

    def __init__(self):
        super().__init__()
        self.fixed = {'s': 1, 'tau': 0, 'b': 8}
        self.group_vars = ['A', 'r', 'v2']
        self.subj_vars = ['v1']

    def tensor_pdf(self, rt, response, test, param):
        tau = param['tau']
        sub_param = param.copy()
        del sub_param['tau']
        return tpdfvar(rt - tau, response, test, **sub_param)

    def function_pdf(self):
        return tpdfvar_func()

    def rvs_test(self, test, param, size):
        rt, response = tpdfvar_rvs(test, **param, size=size)
        return rt, response

    def init_graph(self, rt, response, test):
        data = {'rt': rt, 'response': response, 'test': test}
        with pm.Model() as graph:
            s = 1
            tau = .2
            b = 12
            A = pm.Uniform('A', lower=0, upper=b)
            v1 = pm.Uniform('v1', lower=0, upper=10)
            v2 = pm.Uniform('v2', lower=0, upper=10)
            r = pm.Uniform('r', lower=0, upper=10)
            param = {'A': A, 'b': b, 'v1': v1, 'v2': v2, 'r': r,
                     's': s, 'tau': tau}
            logp = self.tensor_logp(param)
            t_resp = pm.DensityDist('t_resp', logp, observed=data)
        return graph

    def init_graph_hier(self, rt, response, test, subj_idx):
        n_subj = len(np.unique(subj_idx))
        n_trial = np.max(np.unique(subj_idx, return_counts=True)[1])
        data = {'rt': rt, 'response': response, 'test': test,
                'subj_idx': subj_idx}
        with pm.Model() as graph:
            # free parameters
            param = self.fixed.copy()
            A = pm.Uniform('A', lower=0, upper=param['b'])
            v1_mu = pm.HalfNormal('v1_mu', sd=4)
            v1_sd = pm.Gamma('v1_sd', alpha=1.5, beta=0.75)
            v1_offset = pm.Normal('v1_offset', mu=0, sd=1, shape=n_subj)
            v1 = pm.Deterministic('v1', v1_mu + v1_sd * v1_offset)
            v2 = pm.Uniform('v2', lower=-10, upper=10)
            r = pm.Uniform('r', lower=0, upper=1)

            # likelihood function
            param.update(A=A, v1=v1, v2=v2, r=r)
            logp = self.tensor_logp_subj(param, self.subj_vars)
            response = pm.DensityDist('response', logp, observed=data)
        annotate_graph(graph, (n_subj, n_trial))
        return graph


class LBASep(model.Model):
    """Separate process LBA model."""

    def __init__(self):
        super().__init__()
        self.fixed = {'s': 1, 'tau': 0, 'b': 8}
        self.group_vars = ['A', 'v2', 'v4']
        self.subj_vars = ['v1', 'v3']

    def tensor_pdf(self, rt, response, test, param):
        tau = param['tau']
        sub_param = param.copy()
        del sub_param['tau']
        return tpdfsep(rt - tau, response, test, **sub_param)

    def function_pdf(self):
        return tpdfsep_func()

    def rvs_test(self, test, param, size):
        rt, response = tpdfsep_rvs(test, **param, size=size)
        return rt, response

    def init_graph_hier(self, rt, response, test, subj_idx):
        n_subj = len(np.unique(subj_idx))
        n_trial = np.max(np.unique(subj_idx, return_counts=True)[1])
        data = {'rt': rt, 'response': response, 'test': test,
                'subj_idx': subj_idx}
        with pm.Model() as graph:
            # free parameters
            param = self.fixed.copy()
            A = pm.Uniform('A', lower=0, upper=param['b'])
            v1_mu = pm.HalfNormal('v1_mu', sd=4)
            v1_sd = pm.Gamma('v1_sd', alpha=1.5, beta=0.75)
            v1_offset = pm.Normal('v1_offset', mu=0, sd=1, shape=n_subj)
            v1 = pm.Deterministic('v1', v1_mu + v1_sd * v1_offset)

            v2 = pm.Uniform('v2', lower=-10, upper=10)

            v3_mu = pm.HalfNormal('v3_mu', sd=4)
            v3_sd = pm.Gamma('v3_sd', alpha=1.5, beta=0.75)
            v3_offset = pm.Normal('v3_offset', mu=0, sd=1, shape=n_subj)
            v3 = pm.Deterministic('v3', v3_mu + v3_sd * v3_offset)

            v4 = pm.Uniform('v4', lower=-10, upper=10)

            # likelihood function
            param.update(A=A, v1=v1, v2=v2, v3=v3, v4=v4)
            logp = self.tensor_logp_subj(param, self.subj_vars)
            response = pm.DensityDist('response', logp, observed=data)
        annotate_graph(graph, (n_subj, n_trial))
        return graph


class LBANav(model.Model):
    """Dual-process LBA model."""

    def __init__(self):
        super().__init__()
        self.fixed = {'s': 1, 'tau': 0, 'b': 8, 'v4': -10}
        self.group_vars = ['A', 'r', 'v2']
        self.subj_vars = ['v1', 'v3']

    def tensor_pdf(self, rt, response, test, param):
        tau = param['tau']
        sub_param = param.copy()
        del sub_param['tau']
        return tpdfnav(rt - tau, response, test, **sub_param)

    def function_pdf(self):
        return tpdfnav_func()

    def rvs_test(self, test, param, size):
        rt, response = tpdfnav_rvs(test, **param, size=size)
        return rt, response

    def init_graph(self, rt, response, test):
        data = {'rt': rt, 'response': response, 'test': test}
        with pm.Model() as graph:
            # free parameters
            param = self.fixed.copy()
            A = pm.Uniform('A', lower=0, upper=param['b'])
            v1 = pm.Uniform('v1', lower=-10, upper=10)
            v2 = pm.Uniform('v2', lower=-10, upper=10)
            r = pm.Uniform('r', lower=0, upper=1)
            v3 = pm.Uniform('v3', lower=-10, upper=10)
            v4 = pm.Uniform('v4', lower=-10, upper=10)

            # likelihood function
            param.update(A=A, v1=v1, v2=v2, r=r, v3=v3, v4=v4)
            logp = self.tensor_logp(param)
            t_resp = pm.DensityDist('t_resp', logp, observed=data)
        return graph

    def init_graph_hier(self, rt, response, test, subj_idx):
        n_subj = len(np.unique(subj_idx))
        n_trial = np.max(np.unique(subj_idx, return_counts=True)[1])
        data = {'rt': rt, 'response': response, 'test': test,
                'subj_idx': subj_idx}
        with pm.Model() as graph:
            # free parameters
            param = self.fixed.copy()
            A = pm.Uniform('A', lower=0, upper=param['b'])

            v1_mu = pm.HalfNormal('v1_mu', sd=4)
            v1_sd = pm.Gamma('v1_sd', alpha=1.5, beta=0.75)
            v1_offset = pm.Normal('v1_offset', mu=0, sd=1, shape=n_subj)
            v1 = pm.Deterministic('v1', v1_mu + v1_sd * v1_offset)

            v2 = pm.Uniform('v2', lower=-10, upper=10)

            r = pm.Uniform('r', lower=0, upper=1)

            v3_mu = pm.HalfNormal('v3_mu', sd=4)
            v3_sd = pm.Gamma('v3_sd', alpha=1.5, beta=0.75)
            v3_offset = pm.Normal('v3_offset', mu=0, sd=1, shape=n_subj)
            v3 = pm.Deterministic('v3', v3_mu + v3_sd * v3_offset)

            # likelihood function
            param.update(A=A, v1=v1, v2=v2, r=r, v3=v3)
            logp = self.tensor_logp_subj(param, self.subj_vars)
            response = pm.DensityDist('response', logp, observed=data)
        annotate_graph(graph, (n_subj, n_trial))
        return graph

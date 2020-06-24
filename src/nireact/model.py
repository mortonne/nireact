"""Model of behavior in associative inference task."""

import abc
import numpy as np
import scipy.stats as st
import scipy.optimize as optim
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import seaborn as sns
from nireact import task


def log_prob(p):
    # ensure that the probability is not zero before taking log
    eps = 10e-10
    logp = pm.math.log(pm.math.clip(p, eps, np.Inf))
    return logp


def param_search(f_fit, data, bounds, nrep=1, verbose=False):
    """Run a parameter search, with optional replication."""

    f_optim = optim.differential_evolution
    if nrep > 1:
        val = np.zeros(nrep)
        rep = []
        for i in range(nrep):
            if verbose:
                print(f'Starting search {i + 1}/{nrep}...')
            res = f_optim(f_fit, bounds, data, disp=False, tol=.1,
                          init='random')
            rep.append(res)
            val[i] = res['fun']
            if verbose:
                print(f"Final f(x)= {res['fun']:.2f}")
        res = rep[np.argmin(val)]
    else:
        res = f_optim(f_fit, bounds, data, disp=verbose)
    return res


def param_bounds(var_bounds, var_names):
    """Pack group-level parameters."""

    group_lb = [var_bounds[k][0] for k in [*var_names]]
    group_ub = [var_bounds[k][1] for k in [*var_names]]
    bounds = optim.Bounds(group_lb, group_ub)
    return bounds


def subj_bounds(var_bounds, group_vars, subj_vars, n_subj):
    """Pack subject-varying parameters."""

    group_lb = [var_bounds[k][0] for k in [*group_vars]]
    group_ub = [var_bounds[k][1] for k in [*group_vars]]

    subj_lb = np.hstack([np.tile(var_bounds[k][0], n_subj)
                         for k in [*subj_vars]])
    subj_ub = np.hstack([np.tile(var_bounds[k][1], n_subj)
                         for k in [*subj_vars]])
    bounds = optim.Bounds(np.hstack((group_lb, subj_lb)),
                          np.hstack((group_ub, subj_ub)))
    return bounds


def unpack_subj(fixed, x, group_vars, subj_vars):
    """Unpack subject-varying parameters."""

    # unpack group parameters
    param = fixed.copy()
    param.update(dict(zip(group_vars, x)))

    # split up subject-varying parameters
    n_group = len(group_vars)
    xs = x[n_group:]
    if len(xs) % len(subj_vars) != 0:
        raise ValueError('Parameter vector has incorrect length.')
    n_subj = int(len(xs) / len(subj_vars))
    split = [xs[(i * n_subj):(i * n_subj + n_subj)]
             for i in range(len(subj_vars))]

    # construct subject-specific parameters
    subj_param = [dict(zip(subj_vars, pars)) for pars in zip(*split)]
    return param, subj_param


def trace_df(trace):
    """Create a data frame from a trace object."""

    # exclude transformed variables
    var_names = [n for n in trace.varnames if not n.endswith('__')]
    d_var = {var: trace.get_values(var) for var in var_names}
    df = pd.DataFrame(d_var)
    return df


def sample_hier_drift(sd, alpha, beta, size=1):
    """Sample a hierarchical drift parameter."""

    group_mu = st.halfnorm.rvs(sd)
    group_sd = st.gamma.rvs(alpha, 1/beta)
    x = st.norm.rvs(group_mu, group_sd, size)
    return x


def sample_params(fixed, param, subj_param, n_subj):
    """Create a random sample of parameters."""

    d_group = {name: f() for name, f in param.items()}
    d_subj = {name: f() for name, f in subj_param.items()}
    gen_param_subj = [{name: val[i] for name, val in d_subj.items()}
                      for i in range(n_subj)]
    gen_param = fixed.copy()
    gen_param.update(d_group)
    return gen_param, gen_param_subj


def model_gen_fit(model_specs, test, subj_idx, thresh_iqr=5, **sample_kws):
    """Test model fitting for generated data."""

    n_model = len(model_specs)
    n_subj = len(np.unique(subj_idx))
    model_names = [spec['name'] for spec in model_specs]

    # initialize output variables for all comparisons
    comp_vars = ['rank', 'loo', 'p_loo', 'd_loo',
                 'weight', 'se', 'dse', 'warning']
    results = {'winner': np.zeros((n_model, n_model), dtype=int)}
    for var in comp_vars:
        shape = (n_model, n_model)
        if var == 'warning':
            results[var] = np.zeros(shape, dtype=bool)
        elif var == 'rank':
            results[var] = np.zeros(shape, dtype=int)
        else:
            results[var] = np.zeros(shape)

    for i, gen_spec in enumerate(model_specs):
        # generate a parameter set
        param, subj_param = sample_params(gen_spec['fixed'], gen_spec['param'],
                                          gen_spec['subj_param'], n_subj)

        # generate data from the random parameters
        gen_model = gen_spec['model']
        raw = gen_model.gen(test, param, subj_idx, subj_param=subj_param)

        # remove extreme and missing values
        data = task.scrub_rt(raw, thresh_iqr)
        rt = data.rt.values
        response = data.response.values
        samp_test = data.test_type.values
        samp_subj = data.subj_idx.values
        all_trace = {}
        for j, fit_spec in enumerate(model_specs):
            fit_model = fit_spec['model']
            graph = fit_model.init_graph_hier(rt, response,
                                              samp_test, samp_subj)
            trace = pm.sample(model=graph, **sample_kws)
            all_trace[fit_spec['name']] = trace

        # compare models
        df_comp = pm.compare(all_trace, ic='LOO', method='BB-pseudo-BMA',
                             b_samples=10000)

        # save results in correct position
        for j, name in enumerate(model_names):
            for var in comp_vars:
                results[var][i, j] = df_comp.loc[name, var]
            results['winner'][i, j] = 1 if df_comp.loc[name, 'rank'] == 0 else 0
    return results


def model_recovery(model_specs, test, subj_idx, n_rep=1, thresh_iqr=5,
                   **sample_kws):
    """Test model recovery within a set of models."""

    results = {}
    for i in range(n_rep):
        print(f'Replication {i + 1}:')
        rep = model_gen_fit(model_specs, test, subj_idx,
                            thresh_iqr=thresh_iqr, **sample_kws)
        for key, val in rep.items():
            if key not in results:
                results[key] = val
            else:
                results[key] = np.dstack((results[key], val))
    return results


def post_param(trace, fixed, group_vars, subj_vars):
    """Create parameter set from mean of the posterior distribution."""

    param = fixed.copy()
    for name in group_vars:
        param[name] = np.mean(trace.get_values(name))

    m_subj = {name: np.mean(trace.get_values(name), 0) for name in subj_vars}
    n_subj = len(m_subj[subj_vars[0]])
    subj_param = [dict(zip(subj_vars,
                           [m_subj[name][ind] for name in subj_vars]))
                  for ind in range(n_subj)]
    return param, subj_param


def summarize_trace_stats(stats):
    """Summarize trace statistics like ess and rhat."""
    all_var = []
    all_min = []
    all_max = []
    all_med = []
    for key, val in stats.items():
        all_var.append(key)
        if val.size == 1:
            all_min.append(val.values)
            all_max.append(val.values)
            all_med.append(val.values)
        else:
            all_min.append(val.min().values)
            all_max.append(val.max().values)
            all_med.append(val.median().values)
    df = pd.DataFrame({'var': all_var, 'min': all_min, 'med': all_med,
                       'max': all_max})
    return df


def plot_fit(data, sim):
    """Plot fit as a function of test type and response."""

    data.loc[:, 'source'] = 'Data'
    sim.loc[:, 'source'] = 'Model'
    full = pd.concat((data, sim), join='inner', ignore_index=True)
    full.loc[:, 'order'] = full.loc[:, 'response']
    full.loc[full.order == 0, 'order'] = 2
    g = sns.FacetGrid(full, col='test_type', hue='source',
                      row='order', sharey=True)
    g.map(sns.distplot, 'rt', norm_hist=True, kde=False,
          bins=np.linspace(0, 15, 16))
    g.axes[0, 0].legend()
    g.set_titles('')
    g.set_xlabels('Response time (s)')
    g.axes[0, 0].set_ylabel('Correct (relative frequency)')
    g.axes[1, 0].set_ylabel('Incorrect (relative frequency)')
    g.axes[0, 0].set_title('Direct')
    g.axes[0, 1].set_title('Inference')
    return g


def plot_fit_subj(data, sim, test=None):
    """Plot fit by subject."""

    data.loc[:, 'source'] = 'Data'
    sim.loc[:, 'source'] = 'Model'
    full = pd.concat((data, sim), join='inner', ignore_index=True)
    if test is not None:
        full = full.loc[full.test_type == test]

    g = sns.FacetGrid(full, col='subj_idx', col_wrap=5, hue='source', height=2)
    g.map(sns.distplot, 'rt', norm_hist=True, kde=False,
          bins=np.linspace(0, 15, 16))
    g.set_ylabels('Relative frequency')
    g.set_xlabels('Response time (s)')
    g.set_titles('')
    g.axes[0].legend(['Data', 'Model'])
    return g


def plot_fit_scatter(data, sim):
    """Plot mean RT fit by subject."""

    d1 = data.groupby(['subj_idx', 'test_type']).mean()
    d1.loc[:, 'data'] = d1.loc[:, 'rt']
    d2 = sim.groupby(['subj_idx', 'test_type']).mean()
    d2.loc[:, 'model'] = d2.loc[:, 'rt']
    lim = np.max([np.max(d1.rt), np.max(d2.rt)])
    lim += lim / 20
    lim = np.ceil(lim)
    full = pd.concat((d1, d2), axis=1)
    full.reset_index(inplace=True)
    g = sns.FacetGrid(full, col='test_type')
    g.axes[0, 0].plot([0, lim], [0, lim], '-', color='gray')
    g.axes[0, 1].plot([0, lim], [0, lim], '-', color='gray')
    g.map_dataframe(sns.scatterplot, x='data', y='model')
    g.axes[0, 0].set_title('Direct')
    g.axes[0, 1].set_title('Inference')
    g.axes[0, 0].set_aspect('equal', 'box')
    g.axes[0, 1].set_aspect('equal', 'box')
    ticks = g.axes[0, 0].get_yticks()
    g.set_xlabels('Data')
    g.set_ylabels('Model')
    g.axes[0, 0].set_xticks(ticks)
    g.axes[0, 0].set_yticks(ticks)
    g.axes[0, 0].set_xlim(0, lim)
    g.axes[0, 0].set_ylim(0, lim)
    return g


def plot_fit_scatter_comp(data, sim1, sim2, model_names):
    """Plot mean RT fit by subject for two models."""

    d1 = data.groupby(['subj_idx', 'test_type']).mean()
    d1.loc[:, 'data'] = d1.loc[:, 'rt']
    s1 = sim1.groupby(['subj_idx', 'test_type']).mean()
    s1.loc[:, 'model'] = s1.loc[:, 'rt']
    s2 = sim2.groupby(['subj_idx', 'test_type']).mean()
    s2.loc[:, 'model'] = s2.loc[:, 'rt']

    lim = np.max([np.max(d1.rt), np.max(s1.rt), np.max(s2.rt)])
    lim += lim / 20
    lim = np.ceil(lim)
    full1 = pd.concat((d1, s1), axis=1)
    full1.reset_index(inplace=True)
    full2 = pd.concat((d1, s2), axis=1)
    full2.reset_index(inplace=True)
    full = pd.concat([full1, full2], axis=0, keys=model_names)
    full.index = full.index.set_names('model_type', level=0)

    g = sns.FacetGrid(full.reset_index(), row='model_type', col='test_type',
                      height=2.7)
    g.map_dataframe(sns.scatterplot, x='data', y='model')
    ticks = g.axes[0, 0].get_yticks()
    for i in range(2):
        for j in range(2):
            g.axes[i, j].plot([0, lim], [0, lim], '-', color='gray',
                              zorder=0)
            g.axes[i, j].set_aspect('equal', 'box')
            g.axes[i, j].set(xticks=ticks, yticks=ticks,
                             xlim=[0, lim], ylim=[0, lim])
    g.set_xlabels('Response time (s)')
    g.axes[0, 0].set_title('Direct')
    g.axes[0, 1].set_title('Inference')
    g.axes[1, 0].set_title('')
    g.axes[1, 1].set_title('')
    g.axes[0, 0].set_ylabel(model_names[0])
    g.axes[1, 0].set_ylabel(model_names[1])
    return g


class Model:
    """Base class for RT models."""

    def __init__(self):
        self.fixed = None
        self.group_vars = None
        self.subj_vars = None

    @abc.abstractmethod
    def tensor_pdf(self, rt, response, test, param):
        """Probability density function for a set of parameters."""
        pass

    @abc.abstractmethod
    def function_pdf(self):
        """Compiled Theano PDF."""
        pass

    @abc.abstractmethod
    def rvs_test(self, test_type, param, size):
        """Generate responses for a given test type."""
        pass

    def rvs(self, test, param):
        """Generate responses for all test types."""
        n_trial = len(test)
        response = np.zeros(n_trial)
        rt = np.zeros(n_trial)
        test_types = np.unique(test)
        for this_test in test_types:
            ind = test == this_test
            test_rt, test_response = self.rvs_test(this_test, param,
                                                   size=np.count_nonzero(ind))
            response[ind] = test_response
            rt[ind] = test_rt
        return rt, response

    def rvs_subj(self, test, subj_idx, param, subj_param):
        """Generate responses based on subject-varying parameters."""

        unique_idx = np.unique(subj_idx)
        rt = np.zeros(test.shape)
        response = np.zeros(test.shape)
        for idx in unique_idx:
            ind = subj_idx == idx
            param.update(subj_param[idx])
            rt[ind], response[ind] = self.rvs(test[ind], param)
        return rt, response

    def gen(self, test, param, subj_idx=None, nrep=1, subj_param=None):
        """Generate a simulated dataset."""

        data_list = []
        for i in range(nrep):
            if subj_param is not None:
                rt, response = self.rvs_subj(test, subj_idx, param, subj_param)
            else:
                rt, response = self.rvs(test, param)
            rep = pd.DataFrame({'test_type': test, 'rt': rt,
                                'response': response})
            if subj_idx is not None:
                rep.loc[:, 'subj_idx'] = subj_idx
            rep.loc[:, 'rep'] = i
            data_list.append(rep)
        data = pd.concat(data_list, ignore_index=True)
        return data

    def tensor_logp(self, param):
        """Function to evaluate the log PDF for a given response."""
        def logp(rt, response, test):
            p = self.tensor_pdf(rt, response, test, param)
            return log_prob(p)
        return logp

    def tensor_logp_subj(self, param, subj_vars):
        """Function to evaluate the log PDF with subject-varying parameters."""
        def logp(rt, response, test, subj_idx):
            i = tt.cast(subj_idx, 'int64')
            subj_param = param.copy()
            for var in subj_vars:
                subj_param[var] = param[var][i]
            p = self.tensor_pdf(rt, response, test, subj_param)
            return log_prob(p)
        return logp

    def total_logl(self, rt, response, test, param, f_l=None):
        """Calculate log likelihood."""

        if f_l is None:
            f_l = self.function_pdf()
        eps = 0.000001
        # evaluate log likelihood
        lik = f_l(rt, response, test, **param)
        lik[lik < eps] = eps
        logl = np.sum(np.log(lik))
        if np.isnan(logl) or np.isinf(logl):
            return -10e10
        return logl

    def total_logl_subj(self, rt, response, test, subj_idx, param, indiv_param,
                        f_l=None):
        """Calculate log likelihood using subject-varying parameters."""

        if f_l is None:
            f_l = self.function_pdf()

        logl = 0
        for idx, subj_param in enumerate(indiv_param):
            subj_rt = rt[subj_idx == idx]
            subj_response = response[subj_idx == idx]
            subj_test = test[subj_idx == idx]

            param.update(subj_param)
            subj_logl = self.total_logl(subj_rt, subj_response, subj_test,
                                        param, f_l)
            logl += subj_logl
        return logl

    def function_logl(self, fixed, var_names):
        """Generate log likelihood function for use with fitting."""

        param = fixed.copy()
        f_l = self.function_pdf()

        def fit_logl(x, rt, response, test):
            # unpack parameters
            param.update(dict(zip(var_names, x)))
            logl = self.total_logl(rt, response, test, param, f_l)
            return -logl

        return fit_logl

    def function_logl_subj(self, fixed, group_vars, subj_vars):
        """Generate log likelihood function for subject fitting."""

        f_l = self.function_pdf()

        def fit_logl_subj(x, rt, response, test, subj_idx):
            # unpack parameters
            param, subj_param = unpack_subj(fixed, x, group_vars, subj_vars)

            # evaluate all subjects
            logl = self.total_logl_subj(rt, response, test, subj_idx,
                                        param, subj_param, f_l)
            return -logl

        return fit_logl_subj

    def fit(self, rt, response, test, fixed, var_names, var_bounds,
            nrep=1, verbose=False):
        """Estimate maximum likelihood parameters."""

        # maximum likelihood estimation
        fit_logl = self.function_logl(fixed, var_names)
        bounds = param_bounds(var_bounds, var_names)
        data = (rt, response, test)
        res = param_search(fit_logl, data, bounds, nrep=nrep, verbose=verbose)

        # fitted parameters
        param = fixed.copy()
        param.update(dict(zip(var_names, res['x'])))

        # statistics
        logl = -res['fun']
        k = len(var_names)
        n = len(rt)
        bic = np.log(n) * k - 2 * logl
        stats = {'logl': logl, 'k': k, 'n': n, 'bic': bic}

        return param, stats

    def fit_subj(self, rt, response, test, subj_idx,
                 fixed, group_vars, subj_vars, var_bounds,
                 nrep=1, verbose=False):
        """Estimate maximum likelihood parameters for each subject."""

        # maximum likelihood estimation
        fit_logl = self.function_logl_subj(fixed, group_vars, subj_vars)

        # pack parameter bound information
        n_subj = len(np.unique(subj_idx))
        bounds = subj_bounds(var_bounds, group_vars, subj_vars, n_subj)

        data = (rt, response, test, subj_idx)
        res = param_search(fit_logl, data, bounds, nrep=nrep, verbose=verbose)

        # fitted parameters
        param = unpack_subj(fixed, res['x'], group_vars, subj_vars)

        # statistics
        logl = -res['fun']
        k = len(group_vars) + len(subj_vars) * n_subj
        n = len(rt)
        bic = np.log(n) * k - 2 * logl
        stats = {'logl': logl, 'k': k, 'n': n, 'bic': bic}

        return param, stats

    def post_param(self, trace):
        """Get posterior estimates of parameters."""

        param, subj_param = post_param(trace, self.fixed, self.group_vars,
                                       self.subj_vars)
        return param, subj_param

    def dataset_pdf(self, data, param):
        """Evaluate PDF each trial in a dataset."""

        df = data.copy()
        f_pdf = self.function_pdf()
        p = f_pdf(data.rt.values, data.response.values, data.test_type.values,
                  **param)
        df.loc[:, 'p'] = p
        return df

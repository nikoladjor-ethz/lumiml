


def TestInstall():

    from ..base import DeltaBasisFeatures, TriangBasisFeatures, HistBasisFeatures
    from ..models import PoissonElasticNet
    from ..model_selection import PoissonElasticNetCV

    from ..simulator import StretchedExponentialDistribution, Simulator
    import numpy as np

    # generate st_exp distribution
    gamma_eval = np.logspace(-2.5,1,500)
    timeVec = np.linspace(-30, 1000, 10000)

    #### simulator params ###
    bcg_mean = 100
    snr = 1e4
    ## dist params
    bww = 0.5
    tww = 5
    st_exp = StretchedExponentialDistribution(beta_kww=bww, gamma_eval=gamma_eval, n_sum_terms=200, tau_kww=tww)

    sim = Simulator(distribution=st_exp, time_scale=timeVec, background_mean=bcg_mean, snr=snr)
    
    # simulate the streched exponential distribution
    sim.simulate_data()


    dbf = DeltaBasisFeatures(g_min=gamma_eval[0], g_max=gamma_eval[-1], omega=2*np.pi,with_bias=False)
    dbf.fit()

    _filter = sim.time_scale >= 0

    t = sim.time_scale[_filter].copy()
    y = sim.data_simulated.simulated[_filter].copy()

    X = dbf.fit_transform(t[:, np.newaxis])

    penet = PoissonElasticNet(
        alpha=1e-8,
        fix_intercept=True,
        intercept_guess=bcg_mean,
        max_iter=1
    )

    penet_cv = PoissonElasticNetCV(
        estimator=penet,
        param_grid={'alpha': np.logspace(-9, -5, 31)},
        cv=3,
        verbose=1,
        n_jobs=2
    )


    penet_cv.fit(X, y)

    print(penet_cv.best_estimator_.coef_)
    
    return None



if __name__ == '__main__':
    try:
        TestInstall()
    except Exception as e:
        print(e);
        print('Something is wrong with installation! Please read the error message carefuly to try and resolve it.')

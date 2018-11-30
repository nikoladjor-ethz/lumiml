from sklearn.model_selection import GridSearchCV
from lumiml.utils.fileutils import save_to_file
from .models import PoissonElasticNet


class PoissonElasticNetCV(GridSearchCV):
    """
    Helper class for performing grid search cross-validation. This class inherits all properties from
    sklearn.model_selection.GridSearchCV. For information about all other parameters not explained below,
    please refer to GridSearch `documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_

    Parameters
    ----------
    estimator
    param_grid
    scoring
    fit_params
    n_jobs
    iid
    refit
    cv
    verbose
    pre_dispatch
    error_score
    return_train_score

    Attributes
    ------------
    self.best_estimator_

    """
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None, n_jobs=None, iid='warn', refit=True,
                 cv='warn', verbose=0, pre_dispatch='2*n_jobs',
                 error_score='raise-deprecating', return_train_score='warn'):
        """


        """

        super(PoissonElasticNetCV, self).__init__(estimator=estimator, param_grid=param_grid,
                                                  scoring=scoring, fit_params=fit_params, n_jobs=n_jobs, iid=iid,
                                                  refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
                                                  error_score=error_score, return_train_score=return_train_score)

    def save_cross_validation(self, filename, path='./result_log/'):
        """
        Utility method for saving the data obtained using cross-validation.

        Parameters
        ----------
        filename: str
            Name of the file to be saved. By default, extension `.txt` and timestamp will be added to the filename.
        path: path to file
            Path to which to store the data. It can be given as an relative path to the current working directory,
            or defined using `os.path`.

        Returns
        -------
        None
            The stored file will be placed in the path specified

        """
        fields = dir(self)
        field_names_filtered = []
        for elem in fields:
            if not elem.startswith('_') and not callable(self.__getattribute__(elem)):
                field_names_filtered.append(elem)

        field_values = [self.__getattribute__(elem) for elem in field_names_filtered]
        cross_val_data = dict(zip(field_names_filtered, field_values))
        save_to_file(filename=filename, result_df=cross_val_data, result_log_path=path)

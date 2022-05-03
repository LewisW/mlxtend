# Stacking classifier

# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-classifier for stacking
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from sklearn.base import TransformerMixin, clone

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from ._base_classification import _BaseStackingClassifier


class MetaClassifier(_BaseXComposition, _BaseStackingClassifier,
                         TransformerMixin):

    """A meta classifier for scikit-learn estimators for classification.

    Parameters
    ----------
    classifier : object
        Base classifiere.
        Invoking the `fit` method on the `MetaClassifer` will fit clones
        of this original classifier that will
        be stored in the class attribute
        `self.clf_` if `use_clones=True` (default) and
        `fit_base_estimator=True` (default).
    meta_classifier : object
        The meta-classifier to be fitted on the ensemble of
        classifiers
    use_probas : bool (default: False)
        If True, trains meta-classifier based on predicted probabilities
        instead of class labels.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
        - `verbose=2`: Prints info about the parameters of the
                       regressor being fitted
        - `verbose>2`: Changes `verbose` param of the underlying regressor to
           self.verbose - 2
    use_features_in_secondary : bool (default: False)
        If True, the meta-classifier will be trained both on the predictions
        of the original classifiers and the original dataset.
        If False, the meta-classifier will be trained only on the predictions
        of the original classifiers.
    store_train_meta_features : bool (default: False)
        If True, the meta-features computed from the training data used
        for fitting the meta-classifier stored in the
        `self.train_meta_features_` array, which can be
        accessed after calling `fit`.
    use_clones : bool (default: True)
        Clones the classifiers for stacking classification if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Hence, if use_clones=True, the original
        input classifiers will remain unmodified upon using the
        StackingClassifier's `fit` method.
        Setting `use_clones=False` is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.
    fit_base_estimator: bool (default: True)
        Refits classifier in `classifier` if True; uses references to the
        `classifier`, otherwise (assumes that the classifier were
        already fit).
        Note: fit_base_estimators=False will enforce use_clones to be False,
        and is incompatible to most scikit-learn wrappers!
        For instance, if any form of cross-validation is performed
        this would require the re-fitting classifier to training folds, which
        would raise a NotFitterError if fit_base_estimators=False.
        (New in mlxtend v0.6.)

    Attributes
    ----------
    clf_ : list, shape=[n_classifier]
        Fitted classifier (clones of the original classifier)
    meta_clf_ : estimator
        Fitted meta-classifier (clone of the original meta-estimator)
    train_meta_features : numpy array, shape = [n_samples, n_classifier]
        meta-features for training data, where n_samples is the
        number of samples
        in training data and n_classifier is the number of classfier.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
    """

    def __init__(self, classifier, meta_classifier,
                 use_probas=False, verbose=0,
                 use_features_in_secondary=False,
                 store_train_meta_features=False,
                 use_clones=True, fit_base_estimator=True):

        self.classifier = classifier
        self.meta_classifier = meta_classifier
        self.use_probas = use_probas
        self.verbose = verbose
        self.use_features_in_secondary = use_features_in_secondary
        self.store_train_meta_features = store_train_meta_features
        self.use_clones = use_clones
        self.fit_base_estimator = fit_base_estimator

    @property
    def named_classifier(self):
        return _name_estimators([self.classifier])

    def fit(self, X, y, sample_weight=None, clf_kwargs=None, meta_clf_kwargs=None):
        """ Fit ensemble classifer and the meta-classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights passed as sample_weights to each regressor
            in the regressors list as well as the meta_regressor.
            Raises error if some regressor does not support
            sample_weight in the fit() method.

        Returns
        -------
        self : object

        """
        if not self.fit_base_estimator:
            warnings.warn("fit_base_estimators=False "
                          "enforces use_clones to be `False`")
            self.use_clones = False

        if self.use_clones:
            self.clf_ = clone(self.classifier)
            self.meta_clf_ = clone(self.meta_classifier)
        else:
            self.clf_ = self.classifier
            self.meta_clf_ = self.meta_classifier

        if clf_kwargs is None:
            clf_kwargs = {}

        if meta_clf_kwargs is None:
            meta_clf_kwargs = {}

        if self.fit_base_estimator:
            if self.verbose > 0:
                print("Fitting %d classifiers..." % (len(self.classifier)))

            if self.verbose > 0:
                print("Fitting classifier%d: %s" %
                      (i, _name_estimators((self.clf_,))[0][0]))

            if self.verbose > 2:
                if hasattr(self.clf_, 'verbose'):
                    self.clf_.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((self.clf_,))[0][1])
            if sample_weight is None:
                self.clf_.fit(X, y, **clf_kwargs)
            else:
                self.clf_.fit(X, y, sample_weight=sample_weight, **clf_kwargs)

        meta_features = self.predict_meta_features(X)

        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features

        if not self.use_features_in_secondary:
            pass
        elif isinstance(X, pd.DataFrame):
            meta_features = pd.concat([X, pd.DataFrame(meta_features, index=X.index)], axis=1)
        elif sparse.issparse(X):
            meta_features = sparse.hstack((X, meta_features))
        else:
            meta_features = np.hstack((X, meta_features))

        metaY = np.where(self.clf_.predict(X) == y, 1, 0)

        if sample_weight is None:
            self.meta_clf_.fit(meta_features, metaY, **meta_clf_kwargs)
        else:
            self.meta_clf_.fit(meta_features, metaY, sample_weight=sample_weight, **meta_clf_kwargs)

        return self

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        return self._get_params('named_classifier', deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params('classifiers', 'named_classifier', **params)
        return self

    def predict_meta_features(self, X):
        """ Get meta-features of test-data.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Test vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        meta-features : numpy array, shape = [n_samples]
            Returns the meta-features for test data.

        """
        check_is_fitted(self, 'clf_')
        if self.use_probas:
            vals = np.asarray(self.clf_.predict_proba(X))
        else:
            vals = np.asarray(self.clf_.predict(X))

        return vals

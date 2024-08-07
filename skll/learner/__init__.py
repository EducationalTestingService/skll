"""
An easy-to-use class that wraps scikit-learn estimators.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Aoife Cahill (acahill@ets.org)
:organization: ETS
"""

import copy
import logging
from importlib import import_module
from itertools import combinations
from math import floor, log10
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: F401
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction import FeatureHasher
from sklearn.kernel_approximation import (  # noqa: F401
    AdditiveChi2Sampler,
    Nystroem,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LinearRegression,
    LogisticRegression,
    RANSACRegressor,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.linear_model._base import LinearModel
from sklearn.metrics import get_scorer_names, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: F401
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle as sk_shuffle
from sklearn.utils.multiclass import type_of_target

from skll.data import FeatureSet
from skll.data.dict_vectorizer import DictVectorizer
from skll.data.readers import safe_float
from skll.metrics import _CUSTOM_METRICS
from skll.types import (
    CrossValidateTaskResults,
    EvaluateTaskResults,
    FoldMapping,
    IndexIterator,
    LabelType,
    LearningCurveSizes,
    PathOrStr,
)
from skll.utils.constants import (
    CORRELATION_METRICS,
    KNOWN_DEFAULT_PARAM_GRIDS,
    KNOWN_REQUIRES_DENSE,
    MAX_CONCURRENT_PROCESSES,
)

from .utils import (
    Densifier,
    FilteredLeaveOneGroupOut,
    SelectByMinCount,
    _load_learner_from_disk,
    _save_learner_to_disk,
    add_unseen_labels,
    compute_evaluation_metrics,
    compute_num_folds_from_example_counts,
    get_acceptable_classification_metrics,
    get_acceptable_regression_metrics,
    get_predictions,
    load_custom_learner,
    rescaled,
    setup_cv_fold_iterator,
    setup_cv_split_iterator,
    train_and_score,
    write_predictions,
)

# we need a list of learners requiring dense input and a dictionary of
# default parameter grids that we can dynamically update in case we
# import a custom learner
_REQUIRES_DENSE = copy.copy(KNOWN_REQUIRES_DENSE)
_DEFAULT_PARAM_GRIDS = copy.deepcopy(KNOWN_DEFAULT_PARAM_GRIDS)


__all__ = ["Learner", "MAX_CONCURRENT_PROCESSES", "load_custom_learner"]


class Learner(object):
    """
    A simpler interface around scikit-learn classification and regression estimators.

    Parameters
    ----------
    model_type : str
        Name of estimator to create (e.g., ``'LogisticRegression'``).
        See the skll package documentation for valid options.
    probability : bool, default=False
        Should learner return probabilities of all
        labels (instead of just label with highest probability)?
    pipeline : bool, default=False
        Should learner contain a pipeline attribute that
        contains a scikit-learn Pipeline object composed
        of all steps including the vectorizer, the feature
        selector, the sampler, the feature scaler, and the
        actual estimator. Note that this will increase the
        size of the learner object in memory and also when
        it is saved to disk.
    feature_scaling : str, default="none"
        How to scale the features, if at all. Options are
        -  'with_std': scale features using the standard deviation
        -  'with_mean': center features using the mean
        -  'both': do both scaling as well as centering
        -  'none': do neither scaling nor centering
    model_kwargs : Optional[Dict[str, Any]], default=None
        A dictionary of keyword arguments to pass to the
        initializer for the specified model.
    pos_label : Optional[:class:`skll.types.LabelType`], default=None
        An integer or string denoting the label of the class to be
        treated as the positive class in a binary classification
        setting. If ``None``, the class represented by the label
        that appears second when sorted is chosen as the positive
        class. For example, if the two labels in data are "A"
        and "B" and ``pos_label`` is not specified, "B" will
        be chosen as the positive class.
    min_feature_count : int, default=1
        The minimum number of examples a feature
        must have a nonzero value in to be included.
    sampler : Optional[str], default=None
        The sampler to use for kernel approximation, if desired.
        Valid values are
        -  'AdditiveChi2Sampler'
        -  'Nystroem'
        -  'RBFSampler'
        -  'SkewedChi2Sampler'
    sampler_kwargs : Optional[Dict[str, Any]], default=None
        A dictionary of keyword arguments to pass to the
        initializer for the specified sampler.
    custom_learner_path : Optional[str], default=None
        Path to module where a custom classifier is defined.
    logger : Optional[logging.Logger], default=None
        A logging object. If ``None`` is passed, get logger from ``__name__``.

    """

    def __init__(
        self,
        model_type: str,
        probability: bool = False,
        pipeline: bool = False,
        feature_scaling: str = "none",
        model_kwargs: Optional[Dict[str, Any]] = None,
        pos_label: Optional[LabelType] = None,
        min_feature_count: int = 1,
        sampler: Optional[str] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        custom_learner_path: Optional[PathOrStr] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize a learner object with the specified settings."""
        super(Learner, self).__init__()

        self.feat_vectorizer: Optional[Union[DictVectorizer, FeatureHasher]] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_dict: Dict[LabelType, int] = {}
        self.label_list: List[LabelType] = []
        self.pos_label = safe_float(pos_label) if pos_label is not None else pos_label
        self._model = None
        self._store_pipeline = pipeline
        self._feature_scaling = feature_scaling
        self._min_feature_count = min_feature_count
        self.feat_selector: SelectByMinCount = SelectByMinCount(min_count=self._min_feature_count)
        self._model_kwargs: Dict[str, Any] = {}
        self._sampler_kwargs: Dict[str, Any] = {}
        self.logger = logger if logger else logging.getLogger(__name__)

        if model_type not in globals():
            # here, we need to import the custom model and add it
            # to the appropriate lists of models
            globals()[model_type] = load_custom_learner(custom_learner_path, model_type)
            model_class = globals()[model_type]
            default_param_grid = (
                model_class.default_param_grid()
                if hasattr(model_class, "default_param_grid")
                else {}
            )

            # ewww, globals :-(
            global _REQUIRES_DENSE

            _DEFAULT_PARAM_GRIDS.update({model_class: default_param_grid})
            if hasattr(model_class, "requires_dense") and model_class.requires_dense():
                _REQUIRES_DENSE = _REQUIRES_DENSE + (model_class,)

        self._model_type = globals()[model_type]
        # Use setter to set self.probability
        self.probability = probability

        # we need to use dense features under certain conditions:
        # - if we are using any of the estimators that are _known_
        #   to accept only dense features
        # - if we are doing centering as part of feature scaling
        # - if we are using non-negative least squares regression
        self._use_dense_features = (
            issubclass(self._model_type, _REQUIRES_DENSE)
            or self._feature_scaling in {"with_mean", "both"}
            or (
                issubclass(self._model_type, LinearRegression)
                and model_kwargs is not None
                and model_kwargs.get("positive", False)
            )
        )

        # Set default keyword arguments for models that we have some for.
        if issubclass(self._model_type, SVC):
            self._model_kwargs["cache_size"] = 1000
            self._model_kwargs["probability"] = self.probability
            self._model_kwargs["gamma"] = "scale"
            if self.probability:
                self.logger.warning(
                    "Because LibSVM does an internal cross-validation to "
                    "produce probabilities, results will not be exactly "
                    "replicable when using SVC and probability mode."
                )
        elif issubclass(self._model_type, AdaBoostClassifier):
            self._model_kwargs["algorithm"] = "SAMME"
            self._model_kwargs["n_estimators"] = 500
        elif issubclass(
            self._model_type,
            (
                AdaBoostRegressor,
                BaggingClassifier,
                BaggingRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
            ),
        ):
            self._model_kwargs["n_estimators"] = 500
        elif issubclass(self._model_type, DummyClassifier):
            self._model_kwargs["strategy"] = "prior"
        elif issubclass(self._model_type, (LinearSVC, LinearSVR)):
            self._model_kwargs["dual"] = "auto"
        elif issubclass(self._model_type, SVR):
            self._model_kwargs["cache_size"] = 1000
            self._model_kwargs["gamma"] = "scale"
        elif issubclass(self._model_type, SGDClassifier):
            self._model_kwargs["loss"] = "log_loss"
            self._model_kwargs["max_iter"] = 1000
            self._model_kwargs["tol"] = 1e-3
        elif issubclass(self._model_type, SGDRegressor):
            self._model_kwargs["max_iter"] = 1000
            self._model_kwargs["tol"] = 1e-3
        elif issubclass(self._model_type, RANSACRegressor):
            self._model_kwargs["loss"] = "squared_error"
        elif issubclass(self._model_type, (MLPClassifier, MLPRegressor)):
            self._model_kwargs["learning_rate"] = "invscaling"
            self._model_kwargs["max_iter"] = 500
        elif issubclass(self._model_type, LogisticRegression):
            self._model_kwargs["max_iter"] = 1000
            self._model_kwargs["solver"] = "liblinear"
            self._model_kwargs["multi_class"] = "auto"

        if issubclass(
            self._model_type,
            (
                AdaBoostClassifier,
                AdaBoostRegressor,
                BaggingClassifier,
                BaggingRegressor,
                DecisionTreeClassifier,
                DecisionTreeRegressor,
                DummyClassifier,
                ElasticNet,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
                Lasso,
                LinearSVC,
                LinearSVR,
                LogisticRegression,
                MLPClassifier,
                MLPRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
                RANSACRegressor,
                Ridge,
                RidgeClassifier,
                SGDClassifier,
                SGDRegressor,
                SVC,
                TheilSenRegressor,
            ),
        ):
            self._model_kwargs["random_state"] = 123456789

        if sampler_kwargs:
            self._sampler_kwargs.update(sampler_kwargs)
        if sampler:
            sampler_type = globals()[sampler]
            if issubclass(sampler_type, (Nystroem, RBFSampler, SkewedChi2Sampler)):
                self._sampler_kwargs["random_state"] = 123456789
            self.sampler = sampler_type(**self._sampler_kwargs)
        else:
            self.sampler = None

        if model_kwargs:
            # if the model is an AdaBoostClassifier, AdaBoostRegressor,
            # BaggingClassifier, BaggingRegressor, or RANSACRegressor,
            # then we need to convert the specified `estimator` string
            # into an object before passing it in to the learner constructor.
            # We also need to make sure where appropriate, we set the random
            # state to a fixed seed such that results are replicable
            is_ada_has_estimator = (
                issubclass(self._model_type, (AdaBoostRegressor, AdaBoostClassifier))
                and "estimator" in model_kwargs
            )
            is_ransac_has_estimator = (
                issubclass(self._model_type, RANSACRegressor) and "estimator" in model_kwargs
            )
            is_bagging_has_estimator = (
                issubclass(self._model_type, (BaggingClassifier, BaggingRegressor))
                and "estimator" in model_kwargs
            )
            if is_ada_has_estimator or is_ransac_has_estimator or is_bagging_has_estimator:
                base_estimator_kwargs: Dict[str, Any]

                # check if a base estimator name was specified
                base_estimator_name = model_kwargs.get("estimator")

                # set some fixed parameters for specific base estimators
                if base_estimator_name in ["LinearRegression", "MultinomialNB"]:
                    base_estimator_kwargs = {}
                elif base_estimator_name in ["SGDClassifier", "SGDRegressor"]:
                    base_estimator_kwargs = {
                        "max_iter": 1000,
                        "tol": 0.001,
                        "random_state": 123456789,
                    }
                elif base_estimator_name == "SVR":
                    base_estimator_kwargs = {"gamma": "scale"}
                elif base_estimator_name == "SVC":
                    base_estimator_kwargs = {"gamma": "scale", "random_state": 123456789}
                else:
                    base_estimator_kwargs = {"random_state": 123456789}

                # instantiate a base estimator if one was specified and add it
                # to the main learner's model keyword arguments
                if base_estimator_name:
                    base_estimator = globals()[base_estimator_name](**base_estimator_kwargs)
                    model_kwargs["estimator"] = base_estimator

            self._model_kwargs.update(model_kwargs)

    @classmethod
    def from_file(
        cls, learner_path: PathOrStr, logger: Optional[logging.Logger] = None
    ) -> "Learner":
        """
        Load a saved ``Learner`` instance from a file path.

        Parameters
        ----------
        learner_path : :class:`skll.types.PathOrStr`
            The path to a saved ``Learner`` instance file.
        logger : Optional[logging.Logger], default=None
            A logging object. If ``None`` is passed, get logger from ``__name__``.

        Returns
        -------
        :class:`skll.learner.Learner`
            The ``Learner`` instance loaded from the file.

        """
        # use the logger that's passed in or if nothing was passed in,
        # then create a new logger
        logger = logger if logger else logging.getLogger(__name__)

        # call the learner loding utility function
        obj = _load_learner_from_disk(cls, learner_path, logger)
        assert isinstance(obj, cls)
        return obj

    @property
    def model_type(self):
        """Return the model type (i.e., the class)."""
        return self._model_type

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        """Return a dictionary of the underlying scikit-learn model's keyword arguments."""
        return self._model_kwargs

    @property
    def model(self):
        """Return the underlying scikit-learn model."""
        return self._model

    def load(self, learner_path: PathOrStr) -> None:
        """
        Replace the current learner instance with a saved learner.

        Parameters
        ----------
        learner_path : :class:`skll.types.PathOrStr`
            The path to a saved learner object file to load.

        """
        del self.__dict__
        self.__dict__ = Learner.from_file(learner_path).__dict__

    def _convert_coef_array_to_feature_names(self, coef: np.ndarray, feature_name_prefix: str = ""):
        """
        Convert model coefficients array to dictionary.

        Method used by `model_params` to convert the model
        coefficients array into a dictionary with feature names as
        keys and the coefficients as values.

        Parameters
        ----------
        coef : numpy.ndarray
            A numpy array with the model coefficients
        feature_name_prefix : str
            An optional string that should be prefixed to the feature
            name, e.g. the name of the class for LogisticRegression
            or the class pair for SVCs with linear kernels.

        Returns
        -------
        Dict[str, Any]
            A dictionary of labeled weights

        """
        res = {}
        vocabulary = {}

        # if we are doing feature hashing, then we need to make up
        # the feature names
        if isinstance(self.feat_vectorizer, FeatureHasher):
            num_features = len(coef)
            index_width_in_feature_name = int(floor(log10(num_features))) + 1
            feature_names = []
            for idx in range(num_features):
                index_str = str(idx + 1).zfill(index_width_in_feature_name)
                feature_names.append(f"hashed_feature_{index_str}")
            feature_indices = range(num_features)
            vocabulary = dict(zip(feature_names, feature_indices))

        # otherwise we can just use the DictVectorizer vocabulary
        # to get the feature names
        elif isinstance(self.feat_vectorizer, DictVectorizer):
            vocabulary = self.feat_vectorizer.vocabulary_

        # create the final result dictionary with the prefixed
        # feature names and the corresponding coefficient
        for feat, idx in vocabulary.items():
            if coef[idx]:
                res[f"{feature_name_prefix}{feat}"] = coef[idx]

        return res

    @property
    def model_params(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return model parameters (i.e., weights).

        Return the weights for a ``LinearModel`` (e.g., ``Ridge``),
        regression, and liblinear models. If the model was trained using feature
        hashing, then names of the form `hashed_feature_XX` are used instead.

        Returns
        -------
        res : Dict[str, Any]
            A dictionary of labeled weights.
        intercept : Dict[str, Any]
            A dictionary of intercept(s).

        Raises
        ------
        ValueError
            If the instance does not support model parameters.

        """
        res = {}
        intercept = {}
        if (
            isinstance(self._model, LinearModel)
            or (isinstance(self._model, SVR) and self._model.kernel == "linear")
            or isinstance(self._model, SGDRegressor)
        ):
            # also includes RescaledRidge, RescaledSVR, RescaledSGDRegressor

            coef = self.model.coef_
            intercept = {"_intercept_": self.model.intercept_}

            # convert SVR coefficient from a matrix to a 1D array
            # and convert from sparse to dense also if necessary.
            # However, this last bit may not be necessary
            # if we did feature scaling and coef is already dense.
            if isinstance(self._model, SVR):
                if sp.issparse(coef):
                    coef = coef.toarray()
                coef = coef[0]

            # inverse transform to get indices for before feature selection
            coef = coef.reshape(1, -1)
            coef = self.feat_selector.inverse_transform(coef)[0]
            res = self._convert_coef_array_to_feature_names(coef)

        elif isinstance(self._model, LinearSVC) or isinstance(self._model, LogisticRegression):
            label_list = self.label_list

            # if there are only two labels, scikit-learn will only have one
            # set of parameters and they will be associated with label 1 (not
            # 0)
            if len(self.label_list) == 2:
                label_list = self.label_list[-1:]

            if isinstance(self.feat_vectorizer, FeatureHasher):
                self.logger.warning(
                    "No feature names are available since "
                    "this model was trained on hashed "
                    "features."
                )

            for i, label in enumerate(label_list):
                coef = self.model.coef_[i]
                coef = coef.reshape(1, -1)
                coef = self.feat_selector.inverse_transform(coef)[0]
                label_res = self._convert_coef_array_to_feature_names(
                    coef, feature_name_prefix=f"{label}\t"
                )
                res.update(label_res)

            if isinstance(self.model.intercept_, float):
                intercept = {"_intercept_": self.model.intercept_}
            elif self.model.intercept_.any():
                intercept = dict(zip(label_list, self.model.intercept_))  # type: ignore

        # for SVCs with linear kernels, we want to print out the primal
        # weights - that is, the weights for each feature for each one-vs-one
        # binary classifier. These are the weights contained in the `coef_`
        # attribute of the underlying scikit-learn model. This is a matrix that
        # has the shape [(n_classes)*(n_classes -1)/2, n_features] since there
        # are C(n_classes, 2) = n_classes*(n_classes-1)/2 one-vs-one classifiers
        # and each one has weights for each of the features. According to the
        # scikit-learn user guide and the code for the function `_one_vs_one_coef()`
        # in `svm/base.py`, the order of the rows is as follows is "0 vs 1",
        # "0 vs 2", ... "0 vs n", "1 vs 2", "1 vs 3", "1 vs n", ... "n-1 vs n".
        elif isinstance(self._model, SVC) and self._model.kernel == "linear":
            intercept = {}
            if isinstance(self.feat_vectorizer, FeatureHasher):
                self.logger.warning(
                    "No feature names are available since "
                    "this model was trained on hashed "
                    "features."
                )
            for i, class_pair in enumerate(combinations(range(len(self.label_list)), 2)):
                coef = self.model.coef_[i]
                coef = coef.toarray()
                coef = self.feat_selector.inverse_transform(coef)[0]
                class1 = self.label_list[class_pair[0]]
                class2 = self.label_list[class_pair[1]]
                class_pair_res = self._convert_coef_array_to_feature_names(
                    coef, feature_name_prefix=f"{class1}-vs-{class2}\t"
                )
                res.update(class_pair_res)
                intercept[f"{class1}-vs-{class2}"] = self.model.intercept_[i]
        else:
            # not supported
            raise ValueError(
                f"{self._model_type.__name__} is not supported "
                "by model_params with its current settings."
            )

        return res, intercept

    @property
    def probability(self) -> bool:
        """
        Return the value of the probability flag.

        The flag indicages whether the learner return probabilities of all
        labels (instead of just label with highest probability)?
        """
        return self._probability

    @probability.setter
    def probability(self, value: bool) -> None:
        """
        Set the probability flag.

        Parameters
        ----------
        value : bool
            Whether learner should return probabilities of all labels.

        """
        # LinearSVC doesn't support predict_proba
        self._probability = value
        if not hasattr(self.model_type, "predict_proba") and value:
            self.logger.warning(
                "Probability was set to True, but "
                f"{self.model_type.__name__} does not have a "
                "predict_proba() method."
            )
            self._probability = False

    def __getstate__(self) -> Dict[str, Any]:
        """
        Return attributes that should be pickled.

        We need this because we cannot pickle loggers.
        """
        attribute_dict = dict(self.__dict__)
        if "logger" in attribute_dict:
            del attribute_dict["logger"]
        return attribute_dict

    def save(self, learner_path: PathOrStr) -> None:
        """
        Save the ``Learner`` instance to a file.

        Parameters
        ----------
        learner_path : :class:`skll.types.PathOrStr`
            The path to save the ``Learner`` instance to.

        """
        _save_learner_to_disk(self, learner_path)

    def _create_estimator(self):
        """
        Create an estimator.

        Returns
        -------
        estimator
            The estimator that was created.
        default_param_grid : Dict[str, Any]
            The parameter grid for the estimator.

        Raises
        ------
        ValueError
            If there is no default parameter grid for estimator.

        """
        estimator = None
        default_param_grid = None
        for key_class, grid in _DEFAULT_PARAM_GRIDS.items():
            if issubclass(self._model_type, key_class):
                default_param_grid = grid
        if default_param_grid is None:
            raise ValueError(f"{self._model_type.__name__} is not a valid " "learner type.")

        estimator = self._model_type(**self._model_kwargs)

        return estimator, default_param_grid

    def get_feature_names_out(self) -> np.ndarray:
        """
        Return the names of the actual features used by the estimator.

        It is possible for some features to get filtered out by the
        feature selector which means that the vectorizer is no
        longer the correct source for the feature names. This
        method takes into account the feature selector and returns
        the names of the features that were actually selected to be
        used by the estimator.

        Returns
        -------
        names : numpy.ndarray of shape (num_features,)
            Names of features actually used by the estimator.

        Raises
        ------
        ValueError
            If ``self.feat_vectorizer`` is either ``None`` or a
            :class:`sklearn.feature_extraction.FeatureHasher`.

        """
        if isinstance(self.feat_vectorizer, DictVectorizer):
            return self.feat_vectorizer.get_feature_names_out()[self.feat_selector.get_support()]
        else:
            raise ValueError(
                "Cannot get feature names: `feat_vectorizer` is not "
                "defined or a `FeatureHasher`."
            )

    def _check_input_formatting(self, examples: FeatureSet) -> None:
        """
        Check that the examples are properly formatted.

        Parameters
        ----------
        examples : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance to use for training.

        Raises
        ------
        TypeError
            If labels are strings.
        TypeError
            If any features are strings.

        """
        # Make sure the labels for a regression task are not strings.
        if self.model_type._estimator_type == "regressor" and examples.labels is not None:
            for label in examples.labels:
                if isinstance(label, str):
                    raise TypeError(
                        "You are doing regression with string "
                        "labels.  Convert them to integers or "
                        "floats."
                    )

        # make sure that feature values are not strings; to check this
        # we need to get a flattened version of the feature array,
        # whether it is sparse (more likely) or dense
        if examples.features is not None:
            if sp.issparse(examples.features):
                flattened_features = examples.features.data
            else:
                flattened_features = examples.features.flat
            for val in flattened_features:
                if isinstance(val, str):
                    raise TypeError(
                        "You have feature values that are strings. Convert them to floats."
                    )

    def _check_max_feature_value(self, feat_array: np.ndarray):
        """
        Check if the the maximum absolute value of any feature is too large.

        Parameters
        ----------
        feat_array : numpy.ndarray
            A numpy array with features.

        """
        max_feat_abs = np.max(np.abs(feat_array.data))
        if max_feat_abs > 1000.0:
            self.logger.warning(
                "You have a feature with a very large "
                f"absolute value ({max_feat_abs}).  That may "
                "cause the learning algorithm to crash or "
                "perform poorly."
            )

    def _create_label_dict(self, examples: FeatureSet) -> None:
        """
        Create a dictionary of labels for classification problems.

        Parameters
        ----------
        examples : :class:`skll.data.featureset.FeatureSet`
            The examples to use for training.

        """
        # we don't need to do this if we have already done it
        # or for regression models, so simply return.
        if len(self.label_dict) > 0 or self.model_type._estimator_type == "regressor":
            return

        # extract list of unique labels if we are doing classification;
        # note that the output of np.unique() is sorted
        if examples.labels is not None:
            self.label_list = np.unique(examples.labels).tolist()

        # for binary classification, if one label is specified as
        # the positive class, re-sort the label list to make sure
        # that it is last in the list; for multi-class classification
        # raise a warning and set it back to None, since it does not
        # make any sense anyway
        if self.pos_label is not None:
            if len(self.label_list) != 2:
                self.logger.warning(
                    "Ignoring value of `pos_label` for " "multi-class classification."
                )
                self.pos_label = None
            else:
                self.label_list = sorted(self.label_list, key=lambda x: (x == self.pos_label, x))

        # Given a list of all labels in the dataset and a list of the
        # unique labels in the set, convert the first list to an array of
        # numbers.
        self.label_dict = {label: i for i, label in enumerate(self.label_list)}

    def _train_setup(self, examples: FeatureSet) -> None:
        """
        Set up the feature vectorizer and the scaler.

        Parameters
        ----------
        examples : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance to use for training.

        """
        # Check feature values and labels
        self._check_input_formatting(examples)

        # Create feature name -> value mapping
        self.feat_vectorizer = examples.vectorizer

        # Create a scaler if we weren't passed one and we are asked
        # to do feature scaling; note that we do not support feature
        # scaling for `MultinomialNB` learners
        if not issubclass(self._model_type, MultinomialNB) and self._feature_scaling != "none":
            scale_with_mean = self._feature_scaling in {"with_mean", "both"}
            scale_with_std = self._feature_scaling in {"with_std", "both"}
            self.scaler = StandardScaler(
                copy=True, with_mean=scale_with_mean, with_std=scale_with_std
            )
        else:
            # Doing this is to prevent any modification of feature values
            # using a dummy transformation
            self.scaler = StandardScaler(copy=False, with_mean=False, with_std=False)

    def train(
        self,
        examples: FeatureSet,
        param_grid: Optional[Dict[str, Any]] = None,
        grid_search_folds: Union[int, FoldMapping] = 5,
        grid_search: bool = True,
        grid_objective: Optional[str] = None,
        grid_jobs: Optional[int] = None,
        shuffle: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Train model underlying the learner.

        Return the grid search score and a dictionary of grid search results.

        Parameters
        ----------
        examples : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance to use for training.
        param_grid : Optional[Dict[str, Any]], default=None
            The parameter grid to search through for grid
            search. If ``None``, a default parameter grid
            will be used.
        grid_search_folds : Union[int, :class:`skll.types.FoldMapping`], default=5
            The number of folds to use when doing the
            grid search, or a mapping from example IDs to folds.
        grid_search : bool, default=True
            Should we do grid search?
        grid_objective : Optional[str], default=None
            The name of the objective function to use when
            doing the grid search. Must be specified if
            ``grid_search`` is ``True``.
        grid_jobs : Optional[int], default=None
            The number of jobs to run in parallel when doing the
            grid search. If ``None`` or 0, the number of
            grid search folds will be used.
        shuffle : bool, default=False
            Shuffle examples (e.g., for grid search CV.)

        Returns
        -------
        float
            The best grid search objective function score, or 0 if
            we're not doing grid search
        Dict[str, Any]
            Dictionary of grid search CV results with keys such as "params",
            "mean_test_score", etc, that are mapped to lists of values
            associated with each hyperparameter set combination, or
            None if not doing grid search.

        Raises
        ------
        ValueError
            If grid_objective is not a valid grid objective or if
            one is not specified when necessary.
        MemoryError
            If process runs out of memory converting training data to dense.
        ValueError
            If FeatureHasher is used with MultinomialNB.

        """
        # get the estimator type since we need it in multiple places below
        estimator_type = self.model_type._estimator_type

        # if we are asked to do grid search, check that the grid objective
        # is specified and that the specified function is valid for the
        # selected learner
        if grid_search:
            if not grid_objective:
                raise ValueError(
                    "Grid search is on by default. You must "
                    "either specify a grid objective or turn off"
                    " grid search."
                )

            # get the list of objectives that are acceptable in the current
            # prediction scenario and raise an exception if the current
            # objective is not in this allowed list
            if examples.labels is not None:
                label_type = examples.labels.dtype.type
                if estimator_type == "classifier":
                    sorted_unique_labels = np.unique(examples.labels)
                    allowed_objectives = get_acceptable_classification_metrics(sorted_unique_labels)
                else:
                    allowed_objectives = get_acceptable_regression_metrics()

            if grid_objective not in allowed_objectives:
                raise ValueError(
                    f"'{grid_objective}' is not a valid objective"
                    f" function for {self._model_type.__name__} "
                    "with labels of type "
                    f"{label_type.__name__}."
                )

            # If we're using a correlation metric for doing binary
            # classification and probability is set to true, we assume
            # that the user actually wants the `_with_probabilities`
            # version of the metric
            if (
                grid_objective in CORRELATION_METRICS
                and estimator_type == "classifier"
                and self.probability
            ):
                self.logger.info(
                    f'You specified "{grid_objective}" as the '
                    'objective with "probability" set to "true".'
                    " If this is a binary classification task "
                    "with integer labels, the probabilities for "
                    "the positive class will be used to compute "
                    "the correlation."
                )
                old_grid_objective = grid_objective
                new_grid_objective = f"{grid_objective}_probs"
                metrics_module = import_module("skll.metrics")
                metric_func = getattr(metrics_module, "correlation")
                _CUSTOM_METRICS[new_grid_objective] = make_scorer(
                    metric_func, corr_type=grid_objective, response_method="predict_proba"
                )
                grid_objective = new_grid_objective

        # Shuffle so that the folds are random for the inner grid search CV.
        # If grid search is True but shuffle isn't, shuffle anyway.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if grid_search or shuffle:
            if grid_search and not shuffle:
                self.logger.warning(
                    "Training data will be shuffled to randomize "
                    "grid search folds.  Shuffling may yield "
                    "different results compared to scikit-learn."
                )
            ids, labels, features = sk_shuffle(
                examples.ids, examples.labels, examples.features, random_state=123456789
            )
            examples = FeatureSet(
                examples.name, ids, labels=labels, features=features, vectorizer=examples.vectorizer
            )

        # call train setup to set up the vectorizer, the labeldict, and the
        # scaler
        self._create_label_dict(examples)
        self._train_setup(examples)

        # select features
        xtrain = self.feat_selector.fit_transform(examples.features)

        # Convert to dense if necessary
        if self._use_dense_features:
            try:
                xtrain = xtrain.toarray()
            except MemoryError:
                if issubclass(self._model_type, _REQUIRES_DENSE):
                    reason = f"{self._model_type.__name__} does not support " "sparse matrices."
                else:
                    reason = f"{self._feature_scaling} feature scaling " "requires a dense matrix."
                raise MemoryError(
                    "Ran out of memory when converting training"
                    " data to dense. This was required because "
                    f"{reason}"
                )

        if isinstance(self.feat_vectorizer, FeatureHasher) and issubclass(
            self._model_type, MultinomialNB
        ):
            raise ValueError(
                "Cannot use FeatureHasher with MultinomialNB "
                "because MultinomialNB cannot handle negative "
                "feature values."
            )

        # Scale features if necessary
        if self.scaler:
            xtrain = self.scaler.fit_transform(xtrain)

        # check whether any feature values are too large
        self._check_max_feature_value(xtrain)

        # Sampler
        if self.sampler is not None and issubclass(self._model_type, MultinomialNB):
            raise ValueError(
                "Cannot use a sampler with MultinomialNB "
                "because MultinomialNB cannot handle negative "
                "feature values."
            )

        if self.sampler:
            self.logger.warning("Sampler converts sparse matrix to dense")
            if isinstance(self.sampler, SkewedChi2Sampler):
                self.logger.warning("SkewedChi2Sampler uses a dense matrix")
                if sp.issparse(xtrain):
                    xtrain = xtrain.toarray()
            xtrain = self.sampler.fit_transform(xtrain)

        # use label dict transformed version of examples.labels if doing
        # classification
        if examples.labels is not None:
            if estimator_type == "classifier":
                labels = np.array([self.label_dict[label] for label in examples.labels])
            else:
                labels = examples.labels

        # Instantiate an estimator and get the default parameter grid to search
        estimator, default_param_grid = self._create_estimator()

        # Use default parameter grid if we weren't passed one
        # In case the default parameter grid is also empty
        # then there's no point doing the grid search at all
        if grid_search and not param_grid:
            if default_param_grid == {}:
                self.logger.warning(
                    "SKLL has no default parameter grid "
                    "available for the "
                    f"{self._model_type.__name__} learner and"
                    " no parameter grids were supplied. Using"
                    " default values instead of grid search."
                )
                grid_search = False
            else:
                param_grid = default_param_grid

        # set up a grid searcher if we are asked to
        if grid_search:
            # explicitly declare the variable types
            folds: Union[int, IndexIterator]
            final_grid_jobs: int

            # set up grid search folds
            if isinstance(grid_search_folds, int):
                grid_search_folds = compute_num_folds_from_example_counts(
                    grid_search_folds, labels, self.model_type._estimator_type, logger=self.logger
                )

                if not grid_jobs:
                    final_grid_jobs = grid_search_folds
                else:
                    final_grid_jobs = min(grid_search_folds, grid_jobs)
                folds = grid_search_folds
            elif examples.labels is not None:
                # use the number of unique fold IDs as the number of grid jobs
                num_specified_folds = len(set(grid_search_folds.values()))
                if not grid_jobs:
                    final_grid_jobs = num_specified_folds
                else:
                    final_grid_jobs = min(num_specified_folds, grid_jobs)
                # Only retain IDs within folds if they're in grid_search_folds
                dummy_label = next(iter(grid_search_folds.values()))
                fold_groups = [
                    grid_search_folds.get(curr_id, dummy_label) for curr_id in examples.ids
                ]
                kfold = FilteredLeaveOneGroupOut(
                    grid_search_folds, examples.ids, logger=self.logger
                )
                folds = kfold.split(examples.features, examples.labels, fold_groups)

            # limit the number of grid_jobs to be no higher than five or the
            # number of cores for the machine, whichever is lower
            final_grid_jobs = min(final_grid_jobs, cpu_count(), MAX_CONCURRENT_PROCESSES)

            # look up the scorer function in SKLL's custom metrics if the metric
            # is not provided by scikit-learn itself
            assert grid_objective is not None
            final_grid_objective = (
                grid_objective
                if grid_objective in get_scorer_names()
                else _CUSTOM_METRICS[grid_objective]
            )

            # we set `error_score` to "raise" since we want scikit-learn to explicitly
            # raise an exception if the estimator fails to fit for any reason
            grid_searcher = GridSearchCV(
                estimator,
                param_grid,
                scoring=final_grid_objective,
                cv=folds,
                n_jobs=final_grid_jobs,
                error_score="raise",
                pre_dispatch=final_grid_jobs,
            )

            # run the grid search for hyperparameters
            grid_searcher.fit(xtrain, labels)
            self._model = grid_searcher.best_estimator_
            grid_score = grid_searcher.best_score_
            grid_cv_results = grid_searcher.cv_results_
        else:
            self._model = estimator.fit(xtrain, labels)
            grid_score = 0.0
            grid_cv_results = None

        # restore the original of the grid objective if we
        # had futzed with it to handle correlation
        # objectives and probability outputs
        if "old_grid_objective" in locals():
            grid_objective = old_grid_objective
            del _CUSTOM_METRICS[new_grid_objective]

        # store a scikit-learn Pipeline in the `pipeline` attribute
        # composed of a copy of the vectorizer, the selector,
        # the sampler, the scaler, and the estimator. This pipeline
        # attribute can then be used by someone who wants to take a SKLL
        # model and then do further analysis using scikit-learn
        # We are using copies since the user might want to play
        # around with the pipeline and we want to let her do that
        # but keep the SKLL model the same
        if self._store_pipeline:
            # initialize the list that will hold the pipeline steps
            pipeline_steps: List[Tuple[str, Any]] = []

            # start with the vectorizer

            # note that sometimes we may have to end up using dense
            # features or if we were using a SkewedChi2Sampler which
            # requires dense inputs. If this turns out to be the case
            # then let's turn off `sparse` for the vectorizer copy
            # to be stored in the pipeline as well so that it works
            # on the scikit-learn in the same way. However, note that
            # this solution will only work for DictVectorizers. For
            # feature hashers, we manually convert things to dense
            # when we need in SKLL. Therefore, to handle this case,
            # we basically need to create a custom intermediate
            # pipeline stage that will convert the features to dense
            # once the hashing is done since this is what happens
            # in SKLL.
            vectorizer_copy = copy.deepcopy(self.feat_vectorizer)
            if self._use_dense_features or isinstance(self.sampler, SkewedChi2Sampler):
                if isinstance(vectorizer_copy, DictVectorizer):
                    self.logger.warning(
                        "The `sparse` attribute of the DictVectorizer stage "
                        "will be set to `False` in the pipeline since dense "
                        "features are required when centering."
                    )
                    vectorizer_copy.sparse = False
                else:
                    self.logger.warning(
                        "A custom pipeline stage (`Densifier`) will be "
                        "inserted in the pipeline since the current SKLL "
                        "configuration requires dense features."
                    )
                    densifier = Densifier()
                    pipeline_steps.append(("densifier", densifier))
            pipeline_steps.insert(0, ("vectorizer", vectorizer_copy))

            # next add the selector
            pipeline_steps.append(("selector", copy.deepcopy(self.feat_selector)))

            # next, include the scaler
            pipeline_steps.append(("scaler", copy.deepcopy(self.scaler)))

            # next, include the sampler, if there is one
            if self.sampler:
                pipeline_steps.append(("sampler", copy.deepcopy(self.sampler)))

            # finish with the estimator
            pipeline_steps.append(("estimator", copy.deepcopy(self.model)))

            self.pipeline = Pipeline(steps=pipeline_steps)

        return grid_score, grid_cv_results

    def evaluate(
        self,
        examples: FeatureSet,
        prediction_prefix: Optional[str] = None,
        append: bool = False,
        grid_objective: Optional[str] = None,
        output_metrics: List[str] = [],
    ) -> EvaluateTaskResults:
        """
        Evaluate the learner on a given dev or test ``FeatureSet``.

        Parameters
        ----------
        examples : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance to evaluate the performance of the
            model on.
        prediction_prefix : Optional[str], default=None
            If not ``None``, predictions will also be written out to a file with
            the name  ``<prediction_prefix>_predictions.tsv``. Note that
            the prefix can also contain a path.
        append : bool, default=False
            Should we append the current predictions to the file if it exists?
        grid_objective : Optional[str], default=None
            The objective function that was used when doing the grid search.
        output_metrics : List[str], default=[]
            List of additional metric names to compute in addition to grid
            objective.

        Returns
        -------
        :class:`skll.types.EvaluateTaskResults`
            A 6-tuple containing the confusion matrix, the overall accuracy,
            the per-label PRFs, the model parameters, the grid search objective
            function score, and the additional evaluation metrics, if any.
            For regressors, the first two elements in the tuple are ``None``.

        """
        # are we in a regressor or a classifier
        estimator_type = self.model_type._estimator_type

        # make the prediction on the test data; note that these
        # are either class indices or class probabilities
        yhat = self.predict(
            examples, prediction_prefix=prediction_prefix, append=append, class_labels=False
        )

        # for classifiers, convert class labels indices for consistency
        # but account for any unseen labels in the test set that may not
        # have occurred in the training data at all; then get acceptable
        # metrics based on the type of labels we have
        if examples.labels is not None:
            if estimator_type == "classifier":
                sorted_unique_labels = np.unique(examples.labels)
                test_label_list = sorted_unique_labels.tolist()
                train_and_test_label_dict = add_unseen_labels(self.label_dict, test_label_list)
                ytest = np.array([train_and_test_label_dict[label] for label in examples.labels])
                acceptable_metrics = get_acceptable_classification_metrics(sorted_unique_labels)
            # for regressors we do not need to do anything special to the labels
            else:
                train_and_test_label_dict = None
                ytest = examples.labels
                acceptable_metrics = get_acceptable_regression_metrics()

        # check that all of the output metrics are acceptable
        unacceptable_metrics = set(output_metrics).difference(acceptable_metrics)
        if unacceptable_metrics and examples.labels is not None:
            label_type = examples.labels.dtype.type
            raise ValueError(
                "The following metrics are not valid "
                f"for this learner ({self._model_type.__name__})"
                " with these labels of type "
                f"{label_type.__name__}: "
                f"{list(unacceptable_metrics)}"
            )

        # get the values of the evaluation metrics
        (
            conf_matrix,
            accuracy,
            result_dict,
            objective_score,
            metric_scores,
        ) = compute_evaluation_metrics(
            output_metrics,
            ytest,
            yhat,
            estimator_type,
            label_dict=train_and_test_label_dict,
            grid_objective=grid_objective,
            probability=self.probability,
            logger=self.logger,
        )

        # add in the model parameters and return
        model_params: Dict[str, Any] = self.model.get_params()
        res = (conf_matrix, accuracy, result_dict, model_params, objective_score, metric_scores)
        return res

    def predict(
        self,
        examples: FeatureSet,
        prediction_prefix: Optional[str] = None,
        append: bool = False,
        class_labels: bool = True,
    ) -> np.ndarray:
        """
        Generate predictions for the given examples using the learner model.

        Return, and optionally, write out predictions on a given ``FeatureSet``
        to a file. For regressors, the returned and written-out predictions are
        identical. However, for classifiers:

        - if ``class_labels`` is ``True``, class labels are returned
          as well as written out.

        - if ``class_labels`` is ``False`` and the classifier is probabilistic
          (i.e., ``self..probability`` is ``True``), class probabilities are
          returned as well as written out.

        - if ``class_labels`` is ``False`` and the classifier is non-probabilistic
          (i.e., ``self..probability`` is ``False``), class indices are returned
          and class labels are written out.

        TL;DR: for regressors, just ignore ``class_labels``. For classfiers,
        set it to ``True`` to get class labels and ``False`` to get class
        probabilities.

        Parameters
        ----------
        examples : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance to predict labels for.
        prediction_prefix : Optional[str], default=None
            If not ``None``, predictions will also be written out to a file with
            the name  ``<prediction_prefix>_predictions.tsv``. For classifiers,
            the predictions written out are class labels unless the learner is
            probabilistic AND ``class_labels`` is set to ``False``. Note that
            this prefix can also contain a path.
        append : bool, default=False
            Should we append the current predictions to the file if it exists?
        class_labels : bool, default=True
            If ``False``, return either the class probabilities (probabilistic
            classifiers) or the class indices (non-probabilistic ones). If
            ``True``, return the class labels no matter what. Ignored for
            regressors.

        Returns
        -------
        numpy.ndarray
            The predictions returned by the ``Learner`` instance.

        Raises
        ------
        AssertionError
            If invalid predictions are being returned or written out.
        MemoryError
            If process runs out of memory when converting to dense.
        RuntimeError
            If there is a mismatch between the learner vectorizer
            and the test set vectorizer.

        """
        example_ids = examples.ids

        # Need to do some transformations so the features are in the right
        # columns for the test set. Obviously a bit hacky, but storing things
        # in sparse matrices saves memory over our old list of dicts approach.

        # We also need to think about the various combinations of the model
        # vectorizer and the vectorizer for the set for which we want to make
        # predictions:

        # 1. Both vectorizers are DictVectorizers. If they use different sets
        # of features, we raise a warning and transform the features of the
        # prediction set from its space to the trained model space.

        # 2. Both vectorizers are FeatureHashers. If they use different number
        # of feature bins, we should just raise an error since there's no
        # inverse_transform() available for a FeatureHasher - the hash function
        # is not reversible.

        # 3. The model vectorizer is a FeatureHasher but the prediction feature
        # set vectorizer is a DictVectorizer. We should be able to handle this
        # case, since we can just call inverse_transform() on the DictVectorizer
        # and then transform() on the FeatureHasher?

        # 4. The model vectorizer is a DictVectorizer but the prediction feature
        # set vectorizer is a FeatureHasher. Again, we should raise an error here
        # since there's no inverse available for the hasher.

        # 1. both are DictVectorizers
        if isinstance(self.feat_vectorizer, DictVectorizer) and isinstance(
            examples.vectorizer, DictVectorizer
        ):
            if set(self.feat_vectorizer.feature_names_) != set(examples.vectorizer.feature_names_):
                self.logger.warning(
                    "There is mismatch between the training model features "
                    "and the data passed to predict. The prediction features "
                    "will be transformed to the trained model space."
                )
            if self.feat_vectorizer == examples.vectorizer:
                xtest = examples.features
            else:
                xtest = self.feat_vectorizer.transform(
                    examples.vectorizer.inverse_transform(examples.features)
                )

        # 2. both are FeatureHashers
        elif isinstance(self.feat_vectorizer, FeatureHasher) and isinstance(
            examples.vectorizer, FeatureHasher
        ):
            self_feat_vec_tuple = (
                self.feat_vectorizer.dtype,
                self.feat_vectorizer.input_type,
                self.feat_vectorizer.n_features,
            )
            example_feat_vec_tuple = (
                examples.vectorizer.dtype,
                examples.vectorizer.input_type,
                examples.vectorizer.n_features,
            )

            if self_feat_vec_tuple == example_feat_vec_tuple:
                xtest = examples.features
            else:
                self.logger.error(
                    "There is mismatch between the FeatureHasher "
                    "configuration for the training data and the "
                    "configuration for the data passed to predict"
                )
                raise RuntimeError("Mismatched hasher configurations")

        # 3. model is a FeatureHasher and test set is a DictVectorizer
        elif isinstance(self.feat_vectorizer, FeatureHasher) and isinstance(
            examples.vectorizer, DictVectorizer
        ):
            xtest = self.feat_vectorizer.transform(
                examples.vectorizer.inverse_transform(examples.features)
            )

        # 4. model is a DictVectorizer and test set is a FeatureHasher
        elif isinstance(self.feat_vectorizer, DictVectorizer) and isinstance(
            examples.vectorizer, FeatureHasher
        ):
            self.logger.error(
                "Cannot predict with a model using a "
                "DictVectorizer on data that uses a "
                "FeatureHasher"
            )
            raise RuntimeError("Cannot use FeatureHasher for data")

        # filter features based on those selected from training set
        xtest = self.feat_selector.transform(xtest)

        # Convert to dense if necessary
        if self._use_dense_features and not isinstance(xtest, np.ndarray):
            try:
                xtest = xtest.toarray()
            except MemoryError:
                if issubclass(self._model_type, _REQUIRES_DENSE):
                    reason = f"{self._model_type.__name__} does not support " "sparse matrices."
                else:
                    reason = f"{self._feature_scaling} feature scaling " "requires a dense matrix."
                raise MemoryError(
                    "Ran out of memory when converting test "
                    "data to dense. This was required because "
                    f"{reason}"
                )

        # Scale xtest if necessary
        if not issubclass(self._model_type, MultinomialNB) and self.scaler:
            xtest = self.scaler.transform(xtest)

        # Sampler
        if self.sampler:
            self.logger.warning("Sampler converts sparse matrix to dense")
            if isinstance(self.sampler, SkewedChi2Sampler):
                self.logger.warning("SkewedChi2Sampler uses a dense matrix")
                if sp.issparse(xtest):
                    xtest = xtest.toarray()
            xtest = self.sampler.transform(xtest)

        # get the various prediction from this learner on these features
        prediction_dict = get_predictions(self, xtest)

        # for classifiers ...
        if self.model_type._estimator_type == "classifier":
            # return and write class labels if they were explicitly asked for
            if class_labels:
                to_return = to_write = prediction_dict["labels"]
            else:
                # return and write probabilities
                if self.probability:
                    to_return = to_write = prediction_dict["probabilities"]
                # return class indices and write labels
                else:
                    to_return = prediction_dict["raw"]
                    to_write = prediction_dict["labels"]

        # for regressors, it's really simple
        else:
            to_write = to_return = prediction_dict["raw"]

        # check that our predictions to write and return
        # are not invalid; this should NEVER happen
        try:
            assert to_return is not None
            assert to_write is not None
        except AssertionError:
            raise AssertionError("invalid predictions generated")

        # write out the predictions if we are asked to
        if prediction_prefix is not None:
            write_predictions(
                example_ids,
                to_write,
                prediction_prefix,
                self.model_type._estimator_type,
                self.label_list,
                append=append,
            )

        return to_return

    def cross_validate(
        self,
        examples: FeatureSet,
        stratified: bool = True,
        cv_folds: Union[int, FoldMapping] = 10,
        cv_seed: int = 123456789,
        grid_search: bool = True,
        grid_search_folds: Union[int, FoldMapping] = 5,
        grid_jobs: Optional[int] = None,
        grid_objective: Optional[str] = None,
        output_metrics: List[str] = [],
        prediction_prefix: Optional[str] = None,
        param_grid: Optional[Dict[str, Any]] = None,
        shuffle: bool = False,
        save_cv_folds: bool = True,
        save_cv_models: bool = False,
        use_custom_folds_for_grid_search: bool = True,
    ) -> CrossValidateTaskResults:
        """
        Cross-validate the learner on the given training examples.

        Parameters
        ----------
        examples : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance to cross-validate learner performance on.
        stratified : bool, default=True
            Should we stratify the folds to ensure an even
            distribution of labels for each fold?
        cv_folds : Union[int, :class:`skll.types.FoldMapping`], default=10
            The number of folds to use for cross-validation, or
            a mapping from example IDs to folds.
        cv_seed: int, default=123456789
            The value for seeding the random number generator
            used to create the random folds. Note that this
            seed is *only* used if either ``grid_search`` or
            ``shuffle`` are set to ``True``.
        grid_search : bool, default=True
            Should we do grid search when training each fold?
            Note: This will make this take *much* longer.
        grid_search_folds : Union[int, :class:`skll.types.FoldMapping`], default=5
            The number of folds to use when doing the
            grid search, or a mapping from example IDs to folds.
        grid_jobs : Optional[int], default=None
            The number of jobs to run in parallel when doing the
            grid search. If ``None`` or 0, the number of
            grid search folds will be used.
        grid_objective : Optional[str], default=None
            The name of the objective function to use when
            doing the grid search. Must be specified if
            ``grid_search`` is ``True``.
        output_metrics : List[str], default = []
            List of additional metric names to compute in
            addition to the metric used for grid search.
        prediction_prefix : Optional[str], default=None
            If saving the predictions, this is the
            prefix that will be used for the filename.
            It will be followed by ``"_predictions.tsv"``
        param_grid : Optional[Dict[str, Any]], default=None
            The parameter grid to search.
        shuffle : bool, default=False
            Shuffle examples before splitting into folds for CV.
        save_cv_folds : bool, default=True
             Whether to save the cv fold ids or not?
        save_cv_models : bool, default=False
            Whether to save the cv models or not?
        use_custom_folds_for_grid_search : bool, default=True
            If ``cv_folds`` is a custom dictionary, but ``grid_search_folds``
            is not, perhaps due to user oversight, should the same custom
            dictionary automatically be used for the inner grid-search
            cross-validation?

        Returns
        -------
        :class:`skll.types.CrossValidateTaskResults`
           A 5-tuple containing the following:

            List[:class:`skll.types.EvaluateTaskResults`]: the confusion matrix, overall accuracy,
            per-label PRFs, model parameters, objective function score, and
            evaluation metrics (if any) for each fold.

            List[float]: the grid search scores for each fold.

            List[Dict[str, Any]]: list of dictionaries of grid search CV
            results, one per fold, with keys such as "params", "mean_test_score",
            etc, that are mapped to lists of values associated with each
            hyperparameter set combination.

            Optional[:class:`skll.types.FoldMapping`]: dictionary containing the test-fold number
            for each id if ``save_cv_folds`` is ``True``, otherwise ``None``.

            Optional[List[:class:`skll.learner.Learner`]]: list of learners, one for
            each fold if ``save_cv_models`` is ``True``, otherwise ``None``.

        Raises
        ------
        ValueError
            If classification labels are not properly encoded as strings.
        ValueError
            If ``grid_search`` is ``True`` but ``grid_objective`` is ``None``.

        """
        # Seed the random number generator so that randomized
        # algorithms are replicable
        random_state = np.random.RandomState(cv_seed)

        # We need to check whether the labels in the featureset are labels
        # or continuous values. If it's the latter, we need to raise an
        # an exception since the stratified splitting in sklearn does not
        # work with continuous labels. Note that although using random folds
        # _will_ work, we want to raise an error in general since it's better
        # to encode the labels as strings anyway for classification problems.
        if self.model_type._estimator_type == "classifier" and type_of_target(
            examples.labels
        ) not in ["binary", "multiclass"]:
            raise ValueError(
                "Floating point labels must be encoded as " "strings for cross-validation."
            )

        # check that we have an objective since grid search is on by default
        # Note that `train()` would raise this error anyway later but it's
        # better to raise this early on so rather than after a whole bunch of
        # stuff has happened
        if grid_search:
            if not grid_objective:
                raise ValueError(
                    "Grid search is on by default. You must "
                    "either specify a grid objective or turn off"
                    " grid search."
                )

        # Shuffle so that the folds are random for the inner grid search CV.
        # If grid search is True but shuffle isn't, shuffle anyway.
        # You can't shuffle a scipy sparse matrix in place, so unfortunately
        # we make a copy of everything (and then get rid of the old version)
        if grid_search or shuffle:
            if grid_search and not shuffle:
                self.logger.warning(
                    "Training data will be shuffled to "
                    "randomize grid search folds. Shuffling "
                    "may yield different results compared to "
                    "scikit-learn."
                )
            ids, labels, features = sk_shuffle(
                examples.ids, examples.labels, examples.features, random_state=random_state
            )
            examples = FeatureSet(
                examples.name, ids, labels=labels, features=features, vectorizer=examples.vectorizer
            )

        # call train setup
        self._create_label_dict(examples)
        self._train_setup(examples)

        # Set up the cross-validation iterator.
        kfold, cv_groups = setup_cv_fold_iterator(
            cv_folds,
            examples,
            self.model_type._estimator_type,
            stratified=stratified,
            logger=self.logger,
        )

        # When using custom CV folds (a dictionary), if we are planning to do
        # grid search, set the grid search folds to be the same as the custom
        # cv folds unless a flag is set that explicitly tells us not to.
        # Note that this should only happen when we are using the API; otherwise
        # the configparser should take care of this even before this method is called
        if isinstance(cv_folds, dict):
            if grid_search and use_custom_folds_for_grid_search and grid_search_folds != cv_folds:
                self.logger.warning(
                    "The specified custom folds will be used " "for the inner grid search."
                )
                grid_search_folds = cv_folds

        # handle each fold separately & accumulate the predictions and results
        results = []
        grid_search_scores = []
        grid_search_cv_results_dicts = []
        append_predictions = False
        models: Optional[List["Learner"]] = [] if save_cv_models else None
        skll_fold_ids: Optional[FoldMapping] = {} if save_cv_folds else None
        assert examples.labels is not None
        for fold_num, (train_indices, test_indices) in enumerate(
            kfold.split(examples.features, examples.labels, cv_groups)
        ):
            # Train model
            assert examples.labels is not None and examples.features is not None
            self._model = None  # prevent feature vectorizer from being reset.
            train_set = FeatureSet(
                examples.name,
                examples.ids[train_indices],
                labels=examples.labels[train_indices],
                features=examples.features[train_indices],
                vectorizer=examples.vectorizer,
            )

            (grid_search_score, grid_search_cv_results) = self.train(
                train_set,
                grid_search_folds=grid_search_folds,
                grid_search=grid_search,
                grid_objective=grid_objective,
                param_grid=param_grid,
                grid_jobs=grid_jobs,
                shuffle=grid_search,
            )
            grid_search_scores.append(grid_search_score)
            if save_cv_models and models is not None:
                models.append(copy.deepcopy(self))
            grid_search_cv_results_dicts.append(grid_search_cv_results)
            # note: there is no need to shuffle again within each fold,
            # regardless of what the shuffle keyword argument is set to.

            # Evaluate model
            test_tuple = FeatureSet(
                examples.name,
                examples.ids[test_indices],
                labels=examples.labels[test_indices],
                features=examples.features[test_indices],
                vectorizer=examples.vectorizer,
            )
            results.append(
                self.evaluate(
                    test_tuple,
                    prediction_prefix=prediction_prefix,
                    append=append_predictions,
                    grid_objective=grid_objective,
                    output_metrics=output_metrics,
                )
            )
            append_predictions = True

            # save the fold number for each test ID if we were asked to
            if save_cv_folds and skll_fold_ids is not None:
                for index in test_indices:
                    skll_fold_ids[examples.ids[index]] = str(fold_num)

        # return list of results/outputs for all folds
        return (results, grid_search_scores, grid_search_cv_results_dicts, skll_fold_ids, models)

    def learning_curve(
        self,
        examples: FeatureSet,
        metric: str,
        cv_folds: Union[int, FoldMapping] = 10,
        train_sizes: LearningCurveSizes = np.linspace(0.1, 1.0, 5),
        override_minimum: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[int]]:
        """
        Generate learning curves for the learner using the examples.

        The learning curves are generated on the training examples
        via cross-validation. Adapted from the scikit-learn code for learning
        curve generation (cf.``sklearn.model_selection.learning_curve``).

        Parameters
        ----------
        examples : :class:`skll.data.featureset.FeatureSet`
            The ``FeatureSet`` instance to generate the learning curve on.
        cv_folds : Union[int, :class:`skll.types.FoldMapping`], default=10
            The number of folds to use for cross-validation, or a mapping from
            example IDs to folds.
        metric : str
            The name of the metric function to use when computing the train
            and test scores for the learning curve.
        train_sizes : :class:`skll.types.LearningCurveSizes`, default= :func:`numpy.linspace` with start=0.1, stop=1.0, num=5
            Relative or absolute numbers of training examples
            that will be used to generate the learning curve.
            If the type is float, it is regarded as a fraction
            of the maximum size of the training set (that is
            determined by the selected validation method),
            i.e. it has to be within (0, 1]. Otherwise it
            is interpreted as absolute sizes of the training
            sets. Note that for classification the number of
            samples usually have to be big enough to contain
            at least one sample from each class.
        override_minimum : bool, default=False
            Learning curves can be unreliable for very small sizes
            esp. for > 2 labels. If this option is set to ``True``, the
            learning curve would be generated even if the number
            of example is less 500 along with a warning. If ``False``,
            the curve is not generated and an exception is raised instead.

        Returns
        -------
        train_scores : List[float]
            The scores for the training set.
        test_scores : List[float]
            The scores on the test set.
        fit_times : List[float]
            The average times taken to fit each model.
        num_examples : List[int]
            The numbers of training examples used to generate the curve.

        Raises
        ------
        ValueError
            If the number of examples is less than 500.

        """
        # check that the number of training examples is more than the minimum
        # needed for generating a reliable learning curve
        if len(examples) < 500:
            if not override_minimum:
                raise ValueError(
                    f"Number of training examples provided ({len(examples)}) "
                    "is less than the minimum needed (500) for the "
                    "learning curve to be reliable."
                )
            else:
                self.logger.warning(
                    "Learning curves can be unreliable for examples fewer than "
                    f"500. You provided {len(examples)}."
                )

        # raise a warning if we are using a probabilistic classifier
        # since that means we cannot use the predictions directly
        if self.probability:
            self.logger.warning(
                "Since ``probability`` is set, the most likely "
                "class will be computed via an argmax before "
                "computing the curve."
            )

        # Call train setup before since we need to train
        # the learner eventually
        self._create_label_dict(examples)
        self._train_setup(examples)

        # set up the CV split iterator over the train/test featuresets
        # which also returns the maximum number of training examples
        (featureset_iter, n_max_training_samples) = setup_cv_split_iterator(cv_folds, examples)

        # Get the `_translate_train_sizes()` function from scikit-learn
        # since we need it to get the right list of sizes after cross-validation
        _module = import_module("sklearn.model_selection._validation")
        _translate_train_sizes = getattr(_module, "_translate_train_sizes")
        train_sizes_abs = _translate_train_sizes(train_sizes, n_max_training_samples)
        n_unique_ticks = train_sizes_abs.shape[0]

        # Limit the number of parallel jobs for this
        # to be no higher than five or the number of cores
        # for the machine, whichever is lower
        n_jobs = min(cpu_count(), MAX_CONCURRENT_PROCESSES)

        # Run jobs in parallel that train the model on each subset
        # of the training data and compute train and test scores
        parallel = joblib.Parallel(n_jobs=n_jobs, pre_dispatch=n_jobs)
        out = parallel(
            joblib.delayed(train_and_score)(self, train_fs[:n_train_samples], test_fs, metric)
            for train_fs, test_fs in featureset_iter
            for n_train_samples in train_sizes_abs
        )

        # Reshape the outputs
        out = np.array(out)
        n_cv_folds = out.shape[0] // n_unique_ticks
        out = out.reshape(n_cv_folds, n_unique_ticks, 3)
        out = np.asarray(out).transpose((2, 1, 0))

        return list(out[0]), list(out[1]), list(out[2]), list(train_sizes_abs)


# Rescaled regressors
@rescaled
class RescaledBayesianRidge(BayesianRidge):  # noqa: D101
    pass


@rescaled
class RescaledAdaBoostRegressor(AdaBoostRegressor):  # noqa: D101
    pass


@rescaled
class RescaledDecisionTreeRegressor(DecisionTreeRegressor):  # noqa: D101
    pass


@rescaled
class RescaledElasticNet(ElasticNet):  # noqa: D101
    pass


@rescaled
class RescaledGradientBoostingRegressor(GradientBoostingRegressor):  # noqa: D101
    pass


@rescaled
class RescaledHuberRegressor(HuberRegressor):  # noqa: D101
    pass


@rescaled
class RescaledKNeighborsRegressor(KNeighborsRegressor):  # noqa: D101
    pass


@rescaled
class RescaledLars(Lars):  # noqa: D101
    pass


@rescaled
class RescaledLasso(Lasso):  # noqa: D101
    pass


@rescaled
class RescaledLinearRegression(LinearRegression):  # noqa: D101
    pass


@rescaled
class RescaledLinearSVR(LinearSVR):  # noqa: D101
    pass


@rescaled
class RescaledMLPRegressor(MLPRegressor):  # noqa: D101
    pass


@rescaled
class RescaledRandomForestRegressor(RandomForestRegressor):  # noqa: D101
    pass


@rescaled
class RescaledRANSACRegressor(RANSACRegressor):  # noqa: D101
    pass


@rescaled
class RescaledRidge(Ridge):  # noqa: D101
    pass


@rescaled
class RescaledSGDRegressor(SGDRegressor):  # noqa: D101
    pass


@rescaled
class RescaledSVR(SVR):  # noqa: D101
    pass


@rescaled
class RescaledTheilSenRegressor(TheilSenRegressor):  # noqa: D101
    pass

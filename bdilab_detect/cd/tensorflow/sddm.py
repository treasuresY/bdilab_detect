from typing import Optional, Dict, Callable, Union

import numpy as np
import scipy
import pandas as pd
import shap
import sklearn.tree
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
# import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate, Dropout, Conv2D, \
    MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.embeddings import Embedding
from sklearn.tree import DecisionTreeRegressor
from bdilab_detect.cd.base import BaseSDDMDrift
from bdilab_detect.utils.frameworks import Framework
from bdilab_detect.utils.warnings import deprecated_alias

tf.compat.v1.disable_eager_execution()


class SDDMDriftTF(BaseSDDMDrift):
    @deprecated_alias(preprocess_x_ref='preprocess_at_init')
    def __init__(
            self,
            x_ref: np.ndarray,
            y_ref: np.ndarray,
            cnn_model: tf.keras.Model,
            dtr_model: DecisionTreeRegressor,
            p_val: float = .05,
            x_ref_preprocessed: bool = False,
            preprocess_at_init: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            preds_type: str = 'probs',
            binarize_preds: bool = False,
            window_size=50,
            threshold=0.99,
            shap_class=0,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
            **args
    ) -> None:
        """
        Classifier-based drift detector. The classifier is trained on a fraction of the combined
        reference and test data and drift is detected on the remaining data. To use all the data
        to detect drift, a stratified cross-validation scheme can be chosen.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        model
            TensorFlow classification model used for drift detection.
        p_val
            p-value used for the significance of the test.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        preprocess_at_init
            Whether to preprocess the reference data when the detector is instantiated. Otherwise, the reference
            data will be preprocessed at prediction time. Only applies if `x_ref_preprocessed=False`.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs 'probs' or 'logits'.
        binarize_preds
            Whether to test for discrepency on soft (e.g. prob/log-prob) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        reg_loss_fn
            The regularisation term reg_loss_fn(model) is added to the loss function being optimized.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The model preds are then calculated
            on all the out-of-fold predictions. This allows to leverage all the reference and test data
            for drift detection at the expense of longer computation. If both `train_size` and `n_folds`
            are specified, `n_folds` is prioritized.
        retrain_from_scratch
            Whether the classifier should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        seed
            Optional random seed for fold selection.
        optimizer
            Optimizer used during training of the classifier.
        learning_rate
            Learning rate used by optimizer.
        batch_size
            Batch size used during training of the classifier.
        preprocess_batch_fn
            Optional batch preprocessing function. For example to convert a list of objects to a batch which can be
            processed by the model.
        epochs
            Number of training epochs for the classifier for each (optional) fold.
        verbose
            Verbosity level during the training of the classifier.
            0 is silent, 1 a progress bar and 2 prints the statistics after each epoch.
        train_kwargs
            Optional additional kwargs when fitting the classifier.
        dataset
            Dataset object used during training.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            x_ref_preprocessed=x_ref_preprocessed,
            preprocess_at_init=preprocess_at_init,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            preds_type=preds_type,
            binarize_preds=binarize_preds,
            window_size=window_size,
            threshold=threshold,
            input_shape=input_shape,
            data_type=data_type
        )
        if preds_type not in ['probs', 'logits']:
            raise ValueError("'preds_type' should be 'probs' or 'logits'")

        self.y_ref = y_ref
        self.meta.update({'backend': Framework.TENSORFLOW.value})
        x_ref = pd.DataFrame(x_ref)
        y_ref = pd.DataFrame(y_ref)

        self.shap_model = self.explain_model(X_train=x_ref, y_train=y_ref, cnn_model=cnn_model)
        shap = self.shap_model.get_shap(x_ref)
        self.feature_select = self.feat_selection(shap[shap_class])
        if dtr_model is None:
            self.shap_predict = DecisionTreeRegressor()
            self.shap_predict.fit(x_ref, self.shap_model.get_shap(x_ref)[shap_class][:, self.feature_select])
        else:
            self.shap_predict = dtr_model
        self.shap_class = shap_class

        # define and compile classifier model
        # self.original_model = model
        # self.model = clone_model(model)
        # self.loss_fn = BinaryCrossentropy(from_logits=(self.preds_type == 'logits'))
        # self.dataset = partial(dataset, batch_size=batch_size, shuffle=True)
        # self.predict_fn = partial(predict_batch, preprocess_fn=preprocess_batch_fn, batch_size=batch_size)
        # optimizer = optimizer(learning_rate=learning_rate) if isinstance(optimizer, type) else optimizer
        #
        # self.train_kwargs = {'optimizer': optimizer, 'epochs': epochs,
        #                      'reg_loss_fn': reg_loss_fn, 'preprocess_fn': preprocess_batch_fn, 'verbose': verbose}
        # if isinstance(train_kwargs, dict):
        #     self.train_kwargs.update(train_kwargs)

    class explain_model:
        def __init__(self, X_train, y_train, cnn_model=None, layer_name="drift"):
            self.X_train = X_train
            self.y_train = y_train
            self.dtypes = None
            self.model = cnn_model
            self.X_middle_train = None
            self.layer_name = layer_name

            # 进行数据集的归一化处理
            X = X_train
            y = y_train
            self.dtypes = list(zip(X.dtypes.index, map(str, X.dtypes)))
            for k, dtype in self.dtypes:
                if dtype == "float32":
                    X[k] -= X[k].mean()
                    X[k] /= X[k].std()

            # 若整列都可被1整除，则为分类问题
            values = []
            if all(y.iloc[0] % 1 == 0):
                y[0] = y[0].astype(int)
                values = list(set(y[0]))
                if len(values) == 2:
                    data_type = "binary-classifier"
                    y[0] = y[0].replace({values[0]: 0, values[1]: 1})
                elif len(values) < 20:
                    data_type = "multi-classifier"
                    enc = OneHotEncoder()
                    enc.fit(y[0].values.reshape(-1, 1))
                    # one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
                    y = pd.DataFrame(enc.transform(y[0].values.reshape(-1, 1)).toarray())
                else:
                    data_type = "regression"
            else:
                data_type = "regression"

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)

            if self.model is None:
                input_els = []
                encoded_els = []
                for k, dtype in self.dtypes:
                    input_els.append(Input(shape=(1,)))
                    # 从刚刚加入的Input上创建一个嵌入层加平铺层，否则不加入嵌入层
                    if dtype == "int8":
                        e = Flatten()(Embedding(X_train[k].max() + 1, 1)(input_els[-1]))
                    else:
                        e = input_els[-1]
                    encoded_els.append(e)
                encoded_els = concatenate(encoded_els)
                layer1 = Dropout(0.5)(Dense(10, activation="relu", name="input")(encoded_els))
                layer2 = Dense(30, name="drift", activation="relu")(layer1)
                layer3 = Dense(5)(layer2)
                if data_type == "binary-classifier":
                    # train model
                    out = Dense(1, name="output", activation="sigmoid")(layer3)
                    self.model = Model(inputs=input_els, outputs=[out])
                    self.model.compile(optimizer="adam", loss='binary_crossentropy',
                                       metrics=['accuracy'])

                if data_type == "regression":
                    # train model
                    out = Dense(1, name="output")(layer3)
                    self.model = Model(inputs=input_els, outputs=[out])
                    self.model.compile(optimizer="adam", loss='mean_squared_error',
                                       metrics=['accuracy'])
                if data_type == "multi-classifier":
                    # train model
                    out = Dense(len(values), name="output", activation='softmax')(layer3)
                    self.model = Model(inputs=input_els, outputs=[out])
                    self.model.compile(optimizer="adam", loss='categorical_crossentropy',
                                       metrics=['accuracy'])
                self.model.fit(
                    [X_train[k].values for k, t in self.dtypes],
                    y_train,
                    epochs=200,
                    batch_size=512,
                    shuffle=True,
                    validation_data=([X_valid[k].values for k, t in self.dtypes], y_valid),
                    verbose=0,
                    callbacks=[
                        # early_stop
                    ]
                )
            self.explainer = shap.DeepExplainer(
                (self.model.get_layer(self.layer_name).input, self.model.layers[-1].output),
                self.map2layer(X.copy(), self.layer_name))
            self.shap = self.explainer.shap_values(self.map2layer(X, self.layer_name))
            # shap.force_plot(shap_values, X.iloc[299, :])
            self.X_middle_train = self.layer2df(self.X_train)

        def preprocessing(self, x_copy):
            # 同样归一化
            for k, dtype in self.dtypes:
                if dtype == "float32":
                    x_copy[k] -= x_copy[k].mean()
                    x_copy[k] /= x_copy[k].std()
            return x_copy

        def map2layer(self, x, layer_name):
            x_copy = x.copy()
            self.preprocessing(x_copy)

            feed_dict = dict(zip(self.model.inputs, [np.reshape(x_copy[k].values, (-1, 1)) for k, t in self.dtypes]))
            return K.get_session().run(self.model.get_layer(layer_name).input, feed_dict)

        def layer2df(self, x):
            layer = self.map2layer(x, self.layer_name)
            layer = pd.DataFrame(layer)
            return layer

        def get_shap(self, X):
            return self.explainer.shap_values(self.map2layer(X, self.layer_name))

        def re_train(self, X_retrain):
            X_retrain = self.preprocessing(X_retrain)

            # 尝试去更新检测器，实际上是去改变均值，去获取当前均值
            self.explainer = shap.GradientExplainer(
                (self.model.get_layer(self.layer_name).input, self.model.layers[-1].output),
                self.map2layer(X_retrain.copy(), self.layer_name))

    def retrain(self, x):
        self.shap_predict.fit(x, self.shap)

    def score(self, x: np.ndarray) -> Union[np.ndarray, list]:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value, a notion of distance between the trained classifier's out-of-fold performance \
        and that which we'd expect under the null assumption of no drift, \
        and the out-of-fold classifier model prediction probabilities on the reference and test data \
        as well as the associated reference and test instances of the out-of-fold predictions.
        """
        x = pd.DataFrame(x)
        # 预测值
        shap_pred = self.shap_predict.predict(x)
        # 真实值,需要进行过滤
        self.shap = self.shap_model.get_shap(x)[self.shap_class][:, self.feature_select]
        # t_value, p_value = stats.ttest_ind(shap, shap_pred)
        p_value = [scipy.stats.ks_2samp(self.shap[:, i], shap_pred[:, i]).pvalue for i in range(self.shap.shape[1])]
        return p_value

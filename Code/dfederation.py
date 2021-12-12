"""
Decentralized Federated Learning

This module contains the functions for performing decentralized federated
learning on an arbitrary ML model.

"""

from typing import Any, Callable, List, Optional
import copy
from functools import cache

from sklearn import datasets
import torch

import networkx as nx
import numpy as np

_TParameterSetter = Callable[[Any, np.ndarray], None]
_TParameterGetter = Callable[[Any], np.ndarray]


class ParameterGetter:
    """
    Operators for getting the parameters of a model.

    .
    """

    SKLEARN_PARAMETER_GETTER: _TParameterGetter = lambda model: model.coef_
    TORCH_PARAMETER_GETTER: _TParameterGetter = lambda model: model.state_dict()


def _sklearn_parameter_setter(model, parameters):
    model.coef_ = parameters


def _torch_parameter_setter(model, parameters):
    model.load_state_dict(parameters)


class ParameterSetter:
    """
    The following functions are used to set the parameters of the model.

    .
    """

    SKLEARN_PARAMETER_SETTER: _TParameterSetter = _sklearn_parameter_setter
    TORCH_PARAMETER_SETTER: _TParameterSetter = _torch_parameter_setter


_TTrainFunc = Callable[[Any, np.ndarray, np.ndarray, Optional[List]], None]


def _torch_train_func_partial(optimizer, criterion):
    def wrapped(model, x, y, labels=None):
        x = torch.tensor(np.array([x]), requires_grad=True).float().unsqueeze(0)
        y = torch.tensor(np.array(y)).long().unsqueeze(0)

        model.train()
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    return wrapped


class ModelTrainFunc:
    """
    These functions are used to perform one step of training on the model.

    .
    """

    SKLEARN_TRAIN_FUNC: _TTrainFunc = lambda model, x, y, labels: model.partial_fit(
        x, y, labels
    )
    TORCH_TRAIN_FUNC = lambda optimizer, criterion: _torch_train_func_partial(
        optimizer, criterion
    )


_TPredictFunc = Callable[[Any, np.ndarray], np.ndarray]


class ModelPredictFunc:
    """
    These functions are used to perform one step of prediction on the model.

    .
    """

    SKLEARN_PREDICT_FUNC: _TPredictFunc = lambda model, x: model.predict(x)
    TORCH_PREDICT_FUNC: _TPredictFunc = (
        lambda model, x: model(torch.from_numpy(x)).detach().numpy()
    )


_TScoreFunction = Callable[[Any, np.ndarray, np.ndarray], float]


def _torch_mean_accuracy(model, X, y):

    # Run the model on the data:
    yhat = model(torch.from_numpy(X)).detach().numpy()

    # X and y have the same dim0 (batch-size).
    print(X, y)
    return np.mean([X[i] == y[i] for i in range(len(X))])


def _torch_nll_loss(model, xx, yy):
    total_loss = 0
    for x, y in zip(xx, yy):
        x = torch.tensor(np.array([x]), requires_grad=True).float().unsqueeze(0)
        y = torch.tensor(np.array(y)).long().unsqueeze(0)
        model.eval()
        yhat = model(x)
        loss = torch.nn.NLLLoss()(yhat, y)
        total_loss += loss.item()
    return total_loss / len(xx)


class ModelScoreFunc:
    """
    These functions are used to score the model performance.

    .
    """

    SKLEARN_SCORE_FUNC: _TScoreFunction = lambda model, x, y: model.score(x, y)
    TORCH_SCORE_FUNC_MEAN_ACCURACY: _TScoreFunction = _torch_mean_accuracy
    TORCH_SCORE_FUNC_NLL_LOSS: _TScoreFunction = _torch_nll_loss


_TParameterFuser = Callable[[Any, List[Any]], Any]


def _sklearn_parameter_fuser(params, other_params):
    if len(other_params) == 0:
        # this node doesn't have any incoming edges
        return params[:]
    assert (
        params.shape == other_params[0].shape
    ), f"The shapes of the parameters are not equal: \n\tOther: {other_params[0].shape}\n\tMe: {params.shape}"
    # Merge the parameters with your own parameters and average:
    return np.mean([*other_params, params], axis=0)


def _torch_parameter_fuser(params, other_params):
    # params is a state dict, and other_params is a list of state dicts
    # Merge the parameters with your own parameters and average:
    all_params = [*other_params, params]
    return {
        key: torch.mean(torch.stack([param[key] for param in all_params]), dim=0)
        for key in all_params[0].keys()
    }


class ParameterFuser:
    """
    The following functions are used to fuse the parameters of the model.

    .
    """

    SKLEARN_PARAMETER_FUSER: _TParameterFuser = _sklearn_parameter_fuser
    TORCH_PARAMETER_FUSER: _TParameterFuser = _torch_parameter_fuser


class Node:
    """
    A compute-node in the distributed federated learning algorithm graph.

    Each node has a model, and the machinery to train and use it. Consider
    using SKNode and TorchNode for convenience if you are training a model
    with sklearn or torch.

    """

    def __init__(
        self,
        model,
        labels: List = None,
        node_id: int = None,
        parameter_getter: _TParameterGetter = ParameterGetter.SKLEARN_PARAMETER_GETTER,
        parameter_setter: _TParameterSetter = ParameterSetter.SKLEARN_PARAMETER_SETTER,
        model_train_func: _TTrainFunc = ModelTrainFunc.SKLEARN_TRAIN_FUNC,
        model_predict_func: _TPredictFunc = ModelPredictFunc.SKLEARN_PREDICT_FUNC,
        model_score_func: _TScoreFunction = ModelScoreFunc.SKLEARN_SCORE_FUNC,
        parameter_fuser: _TParameterFuser = ParameterFuser.SKLEARN_PARAMETER_FUSER,
    ) -> None:
        """
        Initialize a node.

        Consider using SKNode and TorchNode for convenience if you are training
        a model with sklearn or torch.

        Arguments:
            id (int): The id of the node. If not given, randomly generated.
            model (Model): The model of the node.
            labels (list): The valid labels of the model, if applicable.
            parameter_getter (Callable): A function to get the parameters of the model.
            parameter_setter (Callable): A function to set the parameters of the model.
            model_train_func (Callable): A function to train the model.
            model_predict_func (Callable): A function to predict the labels of the given data.
            model_score_func (Callable): A function to get the score of the model.
            parameter_fuser (Callable): A function to fuse the parameters of the model.

        """
        self.id = node_id if node_id is not None else np.random.randint(0, 100000)
        self.model = copy.deepcopy(model)
        self.labels = labels
        self.parameter_getter: _TParameterGetter = parameter_getter
        self.parameter_setter: _TParameterSetter = parameter_setter
        self.model_train_func: _TTrainFunc = model_train_func
        self.model_predict_func: _TPredictFunc = model_predict_func
        self.model_score_func: _TScoreFunction = model_score_func
        self.parameter_fuser: _TParameterFuser = parameter_fuser

    def __repr__(self) -> str:
        """
        Return a string representation of the node.

        Returns:
            str: A string representation of the node.

        """
        return f"<Node({self.id})>"

    def train_model(self, X_train, y_train) -> None:
        """
        Train the model of the node.

        Arguments:
            X_train: The training data.
            y_train: The training labels.

        Returns:
            None

        """
        return self.model_train_func(self.model, X_train, y_train, self.labels)

    def predict(self, X_test) -> np.ndarray:
        """
        Predict the labels of the given data.

        Arguments:
            X_test: The data to predict.

        Returns:
            np.ndarray: The predicted labels.

        """
        return self.model_predict_func(self.model, X_test)

    def get_score(self, X_test, y_test) -> float:
        """
        Get the score of the node.

        Arguments:
            X_test: The test data.
            y_test: The test labels.

        Returns:
            float: The score of the node.

        """
        return self.model_score_func(self.model, X_test, y_test)

    def get_parameters(self):
        """
        Get the parameters of the model.

        Returns:
            dict: The parameters of the model.

        """
        return self.parameter_getter(self.model)

    def fuse(self, other_nodes: list) -> None:
        """
        Fuse the models of the given nodes.

        Arguments:
            other_nodes: A list of other nodes to fuse from.

        Returns:
            None

        """
        # Get the parameters of the current model
        my_params = self.get_parameters()

        # Get the parameters of the other models
        other_params = [m.get_parameters() for m in other_nodes]

        # Fuse the parameters
        params = self.parameter_fuser(my_params, other_params)

        # Set the parameters of the current model
        self.parameter_setter(self.model, params)

    def get_model(self):
        """
        Get the model of the node.

        Returns:
            linear_model: The model of the node.

        """
        return self.model


class SKNode(Node):
    """
    A convenience-class that wraps Node for scikit-learn models.

    For more information, see the documentation of Node.
    """

    def __init__(self, model, labels: List = None, node_id: int = None, **kwargs):
        """
        Create a Node that uses a scikit-learn-style API.

        Arguments:
            model (Any): A model with a scikit-learn-like API.
            labels (list): The valid labels of the model, if applicable.
            node_id (int): The id of the node. If not given, randomly generated.

        """
        super().__init__(
            model,
            labels,
            node_id,
            parameter_getter=ParameterGetter.SKLEARN_PARAMETER_GETTER,
            parameter_setter=ParameterSetter.SKLEARN_PARAMETER_SETTER,
            model_train_func=ModelTrainFunc.SKLEARN_TRAIN_FUNC,
            model_predict_func=ModelPredictFunc.SKLEARN_PREDICT_FUNC,
            model_score_func=ModelScoreFunc.SKLEARN_SCORE_FUNC,
            parameter_fuser=ParameterFuser.SKLEARN_PARAMETER_FUSER,
            **kwargs,
        )


class TorchNode(Node):
    """
    A convenience-class that wraps Node for torch models.

    For more information, see the documentation of Node.

    """

    def __init__(
        self,
        model,
        labels: List = None,
        node_id: int = None,
        optimizer: torch.optim.Optimizer = None,
        criterion: torch.nn.modules.loss._Loss = None,
        **kwargs,
    ):
        """
        Create a Node that uses a torch-style API.

        Arguments:
            model (Any): A model with a torch-like API.
            labels (list): The valid labels of the model, if applicable.
            node_id (int): The id of the node. If not given, randomly generated.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            criterion (torch.nn.modules.loss._Loss): The loss function for the model.

        """
        super().__init__(
            model,
            labels,
            node_id,
            parameter_getter=kwargs.pop(
                "parameter_getter", ParameterGetter.TORCH_PARAMETER_GETTER
            ),
            parameter_setter=kwargs.pop(
                "parameter_setter", ParameterSetter.TORCH_PARAMETER_SETTER
            ),
            model_train_func=kwargs.pop(
                "model_train_func",
                ModelTrainFunc.TORCH_TRAIN_FUNC(optimizer, criterion),
            ),
            model_predict_func=kwargs.pop(
                "model_predict_func", ModelPredictFunc.TORCH_PREDICT_FUNC
            ),
            model_score_func=kwargs.pop(
                "model_score_func", ModelScoreFunc.TORCH_SCORE_FUNC_MEAN_ACCURACY
            ),
            parameter_fuser=kwargs.pop(
                "parameter_fuser", ParameterFuser.TORCH_PARAMETER_FUSER
            ),
            **kwargs,
        )


class FederatedCommunity:
    """
    A federated community.

    A federated community is a collection of nodes that can be trained and
    evaluated together. The topology of the community is defined by the
    `topology` argument of the constructor, but this can be changed later
    using the `add_edge` and `remove_edge` methods.

    """

    def __init__(self, topology: nx.DiGraph) -> None:
        """
        Initialize a federated community.

        Arguments:
            model: The model to share among the nodes.
            topology: The topology of the community to impose. Nodes here must
                have a "node" attribute of the Node type.

        """
        self._g = topology

    def __repr__(self) -> str:
        """
        Return a string representation of the federated community.

        Returns:
            str: A string representation of the federated community.

        """
        return f"<FederatedCommunity(n={len(self._g)})>"

    def train(self, X_train, y_train, node_id: int) -> None:
        """
        Train a node in the federated community.

        Arguments:
            X_train: The training data.
            y_train: The training labels.
            node_id: The id of the node to train.

        Returns:
            None

        """
        # Get the node
        node = self._g.nodes[node_id]["node"]

        # Train the node
        node.train_model(X_train, y_train)

    def communicate(self, times: int = 1, only_node_ids: list = None) -> None:
        """
        Perform a single communication step.

        If a list of nodes is provided, only those nodes will receive new
        model parameters. Otherwise, all nodes will receive new params.

        Arguments:
            times: The number of times to perform the communication step.
            only_node_ids: A list of nodes that will receive new parameters.

        Returns:
            None

        """
        if times > 1:
            return self.communicate(times=times - 1, only_node_ids=only_node_ids)

        # Get the nodes to communicate with
        nodes = only_node_ids or self._g.nodes

        # Iterate over the receiving-end nodes:
        for node_id in nodes:
            # Fuse this node's model with all of its predecessors' params:
            predecessors = self._g.predecessors(node_id)
            if predecessors:
                self._g.nodes[node_id]["node"].fuse(
                    [self._g.nodes[predecessor]["node"] for predecessor in predecessors]
                )

    def get_scores(self, X_test, y_test) -> dict:
        """
        Get the scores of the nodes in the federated community.

        Arguments:
            X_test: The test data.
            y_test: The test labels.

        Returns:
            dict: A dictionary of node ids and scores.

        """
        return {
            node_id: self._g.nodes[node_id]["node"].get_score(X_test, y_test)
            for node_id in self._g.nodes
        }

    def add_edge(self, source_id, target_id):
        """
        Add an edge to the federated community.

        Arguments:
            source_id: The id of the source node.
            target_id: The id of the target node.

        Returns:
            None

        """
        self._g.add_edge(source_id, target_id)

    def remove_edge(self, source_id, target_id):
        """
        Remove an edge from the federated community.

        Arguments:
            source_id: The id of the source node.
            target_id: The id of the target node.

        Returns:
            None

        """
        self._g.remove_edge(source_id, target_id)

    def get_topology(self):
        """
        Get the topology of the federated community.

        Returns:
            nx.DiGraph: The topology of the federated community.

        """
        return self._g.copy()


@cache
def get_data():
    """
    Get a complete dataset (incl. train and test).

    You can render an example image with:

    ```python
    plt.imshow(get_data()['images'][0])
    ```

    And get the corresponding label with:

    ```python
    get_data()['labels'][0]
    ```

    Arguments:
        None

    Returns:
        sklearn.utils.Bunch: A dictionary-like object containing the data.

    """
    digits = datasets.load_digits()
    return digits


def nth_batch_of_data(num_batches, batch_number, dataset):
    """
    Get the nth batch of data.

    Arguments:
        num_batches: The number of batches to split the data into.
        batch_number: The number of the batch to get.
        dataset: The dataset to get the batch from.

    Returns:
        tuple: A tuple of the batch of data and labels.

    """
    # Get the size of the batches:
    batch_size = len(dataset) // num_batches

    # Get the start and end indices of the batch:
    start_index = batch_number * batch_size
    end_index = start_index + batch_size

    # Get the batch of data and labels:
    X_batch = dataset[int(start_index) : int(end_index)]

    return X_batch

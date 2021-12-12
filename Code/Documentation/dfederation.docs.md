## _Class_ `ParameterGetter`

Operators for getting the parameters of a model.

.

## _Class_ `ParameterSetter`

The following functions are used to set the parameters of the model.

.

## _Class_ `ModelTrainFunc`

These functions are used to perform one step of training on the model.

.

## _Class_ `ModelPredictFunc`

These functions are used to perform one step of prediction on the model.

.

## _Class_ `ModelScoreFunc`

These functions are used to score the model performance.

.

## _Class_ `ParameterFuser`

The following functions are used to fuse the parameters of the model.

.

## _Class_ `Node`

A compute-node in the distributed federated learning algorithm graph.

Each node has a model, and the machinery to train and use it. Consider using SKNode and TorchNode for convenience if you are training a model with sklearn or torch.

## _Function_ `__repr__(self) -> str`

Return a string representation of the node.

### Returns

> -   **str** (`None`: `None`): A string representation of the node.

## _Function_ `train_model(self, X_train, y_train) -> None`

Train the model of the node.

### Arguments

> -   **X_train** (`None`: `None`): The training data.
> -   **y_train** (`None`: `None`): The training labels.

### Returns

    None

## _Function_ `predict(self, X_test) -> np.ndarray`

Predict the labels of the given data.

### Arguments

> -   **X_test** (`None`: `None`): The data to predict.

### Returns

> -   **np.ndarray** (`None`: `None`): The predicted labels.

## _Function_ `get_score(self, X_test, y_test) -> float`

Get the score of the node.

### Arguments

> -   **X_test** (`None`: `None`): The test data.
> -   **y_test** (`None`: `None`): The test labels.

### Returns

> -   **float** (`None`: `None`): The score of the node.

## _Function_ `get_parameters(self)`

Get the parameters of the model.

### Returns

> -   **dict** (`None`: `None`): The parameters of the model.

## _Function_ `fuse(self, other_nodes: list) -> None`

Fuse the models of the given nodes.

### Arguments

> -   **other_nodes** (`None`: `None`): A list of other nodes to fuse from.

### Returns

    None

## _Function_ `get_model(self)`

Get the model of the node.

### Returns

> -   **linear_model** (`None`: `None`): The model of the node.

## _Class_ `SKNode(Node)`

A convenience-class that wraps Node for scikit-learn models.

For more information, see the documentation of Node.

## _Class_ `TorchNode(Node)`

A convenience-class that wraps Node for torch models.

For more information, see the documentation of Node.

## _Class_ `FederatedCommunity`

A federated community.

A federated community is a collection of nodes that can be trained and evaluated together. The topology of the community is defined by the `topology` argument of the constructor, but this can be changed later using the `add_edge` and `remove_edge` methods.

## _Function_ `__init__(self, model, topology: nx.DiGraph) -> None`

Initialize a federated community.

### Arguments

> -   **model** (`None`: `None`): The model to share among the nodes.
> -   **topology** (`None`: `None`): The topology of the community to impose. Nodes here must have a "node" attribute of the Node type.

## _Function_ `__repr__(self) -> str`

Return a string representation of the federated community.

### Returns

> -   **str** (`None`: `None`): A string representation of the federated community.

## _Function_ `train(self, X_train, y_train, node_id: int) -> None`

Train a node in the federated community.

### Arguments

> -   **X_train** (`None`: `None`): The training data.
> -   **y_train** (`None`: `None`): The training labels.
> -   **node_id** (`None`: `None`): The id of the node to train.

### Returns

    None

## _Function_ `communicate(self, times: int = 1, only_node_ids: list = None) -> None`

Perform a single communication step.

If a list of nodes is provided, only those nodes will receive new model parameters. Otherwise, all nodes will receive new params.

### Arguments

> -   **times** (`None`: `None`): The number of times to perform the communication step.
> -   **only_node_ids** (`None`: `None`): A list of nodes that will receive new parameters.

### Returns

    None

## _Function_ `get_scores(self, X_test, y_test) -> dict`

Get the scores of the nodes in the federated community.

### Arguments

> -   **X_test** (`None`: `None`): The test data.
> -   **y_test** (`None`: `None`): The test labels.

### Returns

> -   **dict** (`None`: `None`): A dictionary of node ids and scores.

## _Function_ `add_edge(self, source_id, target_id)`

Add an edge to the federated community.

### Arguments

> -   **source_id** (`None`: `None`): The id of the source node.
> -   **target_id** (`None`: `None`): The id of the target node.

### Returns

    None

## _Function_ `remove_edge(self, source_id, target_id)`

Remove an edge from the federated community.

### Arguments

> -   **source_id** (`None`: `None`): The id of the source node.
> -   **target_id** (`None`: `None`): The id of the target node.

### Returns

    None

## _Function_ `get_topology(self)`

Get the topology of the federated community.

### Returns

> -   **nx.DiGraph** (`None`: `None`): The topology of the federated community.

## _Function_ `get_data()`

Get a complete dataset (incl. train and test).

### Arguments

    None

### Returns

> -   **sklearn.utils.Bunch** (`None`: `None`): A dictionary-like object containing the data.

## _Function_ `nth_batch_of_data(num_batches, batch_number, dataset)`

Get the nth batch of data.

### Arguments

> -   **num_batches** (`None`: `None`): The number of batches to split the data into.
> -   **batch_number** (`None`: `None`): The number of the batch to get.
> -   **dataset** (`None`: `None`): The dataset to get the batch from.

### Returns

> -   **tuple** (`None`: `None`): A tuple of the batch of data and labels.

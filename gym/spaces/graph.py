"""Implementation of a space that represents graph information where nodes and edges can be represented with euclidean space."""
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from gym.logger import warn
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.space import Space


class GraphInstance(NamedTuple):
    """A Graph space instance.

    * nodes (np.ndarray): an (n x ...) sized array representing the features for n nodes, (...) must adhere to the shape of the node space.
    * edges (Optional[np.ndarray]): an (m x ...) sized array representing the features for m edges, (...) must adhere to the shape of the edge space.
    * edge_links (Optional[np.ndarray]): an (m x 2) sized array of ints representing the indices of the two nodes that each edge connects.
    """

    nodes: np.ndarray
    edges: Optional[np.ndarray]
    edge_links: Optional[np.ndarray]


class Graph(Space):
    r"""A space representing graph information as a series of `nodes` connected with `edges` according to an adjacency matrix represented as a series of `edge_links`.

    Example usage::

        self.observation_space = spaces.Graph(node_space=space.Box(low=-100, high=100, shape=(3,)), edge_space=spaces.Discrete(3))
    """

    def __init__(
        self,
        node_space: Union[Box, Discrete],
        edge_space: Union[None, Box, Discrete],
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        r"""Constructor of :class:`Graph`.

        The argument ``node_space`` specifies the base space that each node feature will use.
        This argument must be either a Box or Discrete instance.

        The argument ``edge_space`` specifies the base space that each edge feature will use.
        This argument must be either a None, Box or Discrete instance.

        Args:
            node_space (Union[Box, Discrete]): space of the node features.
            edge_space (Union[None, Box, Discrete]): space of the node features.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
        """
        assert isinstance(
            node_space, (Box, Discrete)
        ), f"Values of the node_space should be instances of Box or Discrete, got {type(node_space)}"
        if edge_space is not None:
            assert isinstance(
                edge_space, (Box, Discrete)
            ), f"Values of the edge_space should be instances of None Box or Discrete, got {type(node_space)}"

        self.node_space = node_space
        self.edge_space = edge_space

        super().__init__(None, None, seed)

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return False

    def _generate_sample_space(
        self, base_space: Union[None, Box, Discrete], num: int
    ) -> Optional[Union[Box, MultiDiscrete]]:
        if num == 0 or base_space is None:
            return None

        if isinstance(base_space, Box):
            return Box(
                low=np.array(max(1, num) * [base_space.low]),
                high=np.array(max(1, num) * [base_space.high]),
                shape=(num,) + base_space.shape,
                dtype=base_space.dtype,
                seed=self.np_random,
            )
        elif isinstance(base_space, Discrete):
            return MultiDiscrete(nvec=[base_space.n] * num, seed=self.np_random)
        else:
            raise TypeError(
                f"Expects base space to be Box and Discrete, actual space: {type(base_space)}."
            )

    def sample(
        self,
        mask: Optional[
            Tuple[
                Optional[Union[np.ndarray, tuple]],
                Optional[Union[np.ndarray, tuple]],
            ]
        ] = None,
        num_nodes: int = 10,
        num_edges: Optional[int] = None,
    ) -> GraphInstance:
        """Generates a single sample graph with num_nodes between 1 and 10 sampled from the Graph.

        Args:
            mask: An optional tuple of optional node and edge mask that is only possible with Discrete spaces
                (Box spaces don't support sample masks).
                If no `num_edges` is provided then the `edge_mask` is multiplied by the number of edges
            num_nodes: The number of nodes that will be sampled, the default is 10 nodes
            num_edges: An optional number of edges, otherwise, a random number between 0 and `num_nodes`^2

        Returns:
            A NamedTuple representing a graph with attributes .nodes, .edges, and .edge_links.
        """
        assert (
            num_nodes > 0
        ), f"The number of nodes is expected to be greater than 0, actual value: {num_nodes}"

        if mask is not None:
            node_space_mask, edge_space_mask = mask
        else:
            node_space_mask, edge_space_mask = None, None

        # we only have edges when we have at least 2 nodes
        if num_edges is None:
            if num_nodes > 1:
                # maximal number of edges is `n*(n-1)` allowing self connections and two-way is allowed
                num_edges = self.np_random.integers(num_nodes * (num_nodes - 1))
            else:
                num_edges = 0

            if edge_space_mask is not None:
                edge_space_mask = tuple(edge_space_mask for _ in range(num_edges))
        else:
            if self.edge_space is None:
                warn(
                    f"The number of edges is set ({num_edges}) but the edge space is None."
                )
            assert (
                num_edges >= 0
            ), f"Expects the number of edges to be greater than 0, actual value: {num_edges}"
        assert num_edges is not None

        sampled_node_space = self._generate_sample_space(self.node_space, num_nodes)
        sampled_edge_space = self._generate_sample_space(self.edge_space, num_edges)

        assert sampled_node_space is not None
        sampled_nodes = sampled_node_space.sample(node_space_mask)
        sampled_edges = (
            sampled_edge_space.sample(edge_space_mask)
            if sampled_edge_space is not None
            else None
        )

        sampled_edge_links = None
        if sampled_edges is not None and num_edges > 0:
            sampled_edge_links = self.np_random.integers(
                low=0, high=num_nodes, size=(num_edges, 2)
            )

        return GraphInstance(sampled_nodes, sampled_edges, sampled_edge_links)

    def contains(self, x: GraphInstance) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, GraphInstance):
            # Checks the nodes
            if isinstance(x.nodes, np.ndarray):
                if all(node in self.node_space for node in x.nodes):
                    # Check the edges and edge links which are optional
                    if isinstance(x.edges, np.ndarray) and isinstance(
                        x.edge_links, np.ndarray
                    ):
                        assert x.edges is not None
                        assert x.edge_links is not None
                        if self.edge_space is not None:
                            if all(edge in self.edge_space for edge in x.edges):
                                if np.issubdtype(x.edge_links.dtype, np.integer):
                                    if x.edge_links.shape == (len(x.edges), 2):
                                        if np.all(
                                            np.logical_and(
                                                x.edge_links >= 0,
                                                x.edge_links < len(x.nodes),
                                            )
                                        ):
                                            return True
                    else:
                        return x.edges is None and x.edge_links is None
        return False

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include node_space and edge_space

        Returns:
            A representation of the space
        """
        return f"Graph({self.node_space}, {self.edge_space})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return (
            isinstance(other, Graph)
            and (self.node_space == other.node_space)
            and (self.edge_space == other.edge_space)
        )

    def to_jsonable(self, sample_n: NamedTuple) -> list:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as list of dicts
        ret_n = []
        for sample in sample_n:
            ret = {}
            ret["nodes"] = sample.nodes.tolist()
            if sample.edges is not None:
                ret["edges"] = sample.edges.tolist()
                ret["edge_links"] = sample.edge_links.tolist()
            ret_n.append(ret)
        return ret_n

    def from_jsonable(self, sample_n: Sequence[dict]) -> list:
        """Convert a JSONable data type to a batch of samples from this space."""
        ret = []
        for sample in sample_n:
            if "edges" in sample:
                ret_n = GraphInstance(
                    np.asarray(sample["nodes"]),
                    np.asarray(sample["edges"]),
                    np.asarray(sample["edge_links"]),
                )
            else:
                ret_n = GraphInstance(
                    np.asarray(sample["nodes"]),
                    None,
                    None,
                )
            ret.append(ret_n)
        return ret

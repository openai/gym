"""Implementation of a space that represents graph information where nodes and edges can be represented with euclidean space."""
from collections import namedtuple
from typing import NamedTuple, Optional, Sequence, Union

import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.space import Space
from gym.utils import seeding


class GraphInstance(namedtuple("GraphInstance", ["nodes", "edges", "edge_links"])):
    r"""Returns a NamedTuple representing a graph object.

    Args:
        nodes (np.ndarray): an (n x ...) sized array representing the features for n nodes.
        (...) must adhere to the shape of the node space.

        edges (np.ndarray): an (m x ...) sized array representing the features for m nodes.
        (...) must adhere to the shape of the edge space.

        edge_links (np.ndarray): an (m x 2) sized array of ints representing the two nodes that each edge connects.

    Returns:
        A NamedTuple representing a graph with `.nodes`, `.edges`, and `.edge_links`.
    """


class Graph(Space):
    r"""A space representing graph information as a series of `nodes` connected with `edges` according to an adjacency matrix represented as a series of `edge_links`.

    Example usage::

        self.observation_space = spaces.Graph(node_space=space.Box(low=-100, high=100, shape=(3,)), edge_space=spaces.Discrete(3))
    """

    def __init__(
        self,
        node_space: Union[Box, Discrete],
        edge_space: Union[None, Box, Discrete],
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
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

    def _generate_sample_space(
        self, base_space: Union[None, Box, Discrete], num: int
    ) -> Optional[Union[Box, Discrete]]:
        # the possibility of this space , got {type(base_space)}aving nothing
        if num == 0:
            return None

        if isinstance(base_space, Box):
            return Box(
                low=np.array(max(1, num) * [base_space.low]),
                high=np.array(max(1, num) * [base_space.high]),
                shape=(num, *base_space.shape),
                dtype=base_space.dtype,
                seed=self._np_random,
            )
        elif isinstance(base_space, Discrete):
            return MultiDiscrete(nvec=[base_space.n] * num, seed=self._np_random)
        elif base_space is None:
            return None
        else:
            raise AssertionError(
                f"Only Box and Discrete can be accepted as a base_space, got {type(base_space)}, you should not have gotten this error."
            )

    def _sample_sample_space(self, sample_space) -> Optional[np.ndarray]:
        if sample_space is not None:
            return sample_space.sample()
        else:
            return None

    def sample(self) -> NamedTuple:
        """Generates a single sample graph with num_nodes between 1 and 10 sampled from the Graph.

        Returns:
            A NamedTuple representing a graph with attributes .nodes, .edges, and .edge_links.
        """
        num_nodes = self.np_random.integers(low=1, high=10)

        # we only have edges when we have at least 2 nodes
        num_edges = 0
        if num_nodes > 1:
            # maximal number of edges is (n*n) allowing self connections and two way is allowed
            num_edges = self.np_random.integers(num_nodes * num_nodes)

        node_sample_space = self._generate_sample_space(self.node_space, num_nodes)
        edge_sample_space = self._generate_sample_space(self.edge_space, num_edges)

        sampled_nodes = self._sample_sample_space(node_sample_space)
        sampled_edges = self._sample_sample_space(edge_sample_space)

        sampled_edge_links = None
        if sampled_edges is not None and num_edges > 0:
            sampled_edge_links = self.np_random.integers(
                low=0, high=num_nodes, size=(num_edges, 2)
            )

        return GraphInstance(sampled_nodes, sampled_edges, sampled_edge_links)

    def contains(self, x: GraphInstance) -> bool:
        """Return boolean specifying if x is a valid member of this space.

        Returns False when:
        - any node in nodes is not contained in Graph.node_space
        - edge_links is not of dtype int
        - len(edge_links) != len(edges)
        - has edges but Graph.edge_space is None
        - edge_links has index less than 0
        - edge_links has index more than number of nodes
        - any edge in edges is not contained in Graph.edge_space
        """
        if not isinstance(x, GraphInstance):
            return False
        if x.edges is not None:
            if not np.issubdtype(x.edge_links.dtype, np.integer):
                return False
            if x.edge_links.shape[-1] != 2:
                return False
            if self.edge_space is None:
                return False
            if x.edge_links.min() < 0:
                return False
            if x.edge_links.max() >= len(x.nodes):
                return False
            if len(x.edges) != len(x.edge_links):
                return False
            if any(edge not in self.edge_space for edge in x.edges):
                return False
        if any(node not in self.node_space for node in x.nodes):
            return False

        return True

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

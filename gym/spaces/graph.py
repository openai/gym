from typing import Optional, Sequence, Union

import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.space import Space
from gym.utils import seeding


class GraphObj:
    r"""A base for constructing information as graphs."""
    def __init__(
        self,
        nodes: np.ndarray,
        edges: Optional[np.ndarray] = None,
        edge_links: Optional[np.ndarray] = None,
    ):
        r"""Constructor for Graph information.

        ``nodes`` must be a nx... sized vector, where ... denotes the shape of
        the base shape that each node feature must be.

        ``edges`` must be either None or a np.ndarray where the first
        dimension (denoted n) is the number of edges.
        If edges is None, then edge_links must also be None.

        ``edge_links`` must be a nx2 sized array of ints, where edge_links.max()
        is not to be equal or larger than the size of the first dimension of nodes, and
        edge_links.min() is not to be smaller than 0.
        """
        self.nodes = nodes
        self.edges = edges
        self.edge_links = edge_links

    def __repr__(self) -> str:
        """A string representation of this graph.

        The representation will include nodes, edges, and edge_links

        Returns:
            The information in this graph.
        """
        return f"GraphObj(nodes: \n{self.nodes}, \n\nedges: \n{self.edges}, \n\nedge_links\n{self.edge_links})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance."""
        if not isinstance(other, GraphObj):
            return False
        if np.any(self.nodes != other.nodes):
            return False
        if self.edges is not None:
            if other.edges is None:
                print('fail1')
                return False
            if np.all(self.edges != other.edges):
                print('fail2')
                return False
            if other.edge_links is None:
                print('fail3')
                return False
            if np.all(self.edge_links != other.edge_links):
                print('fail4')
                return False
        else:
            if other.edges is not None:
                print('fail6')
                return False
            if other.edge_links is not None:
                print('fail7')
                return False

        return True


class Graph(Space):
    r"""A dictionary representing graph spaces with `node_features`, `edge_features` and `edge_links`.

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
        This argument must be either a Box or Discrete instance.

        Args:
            node_space (Union[Box, Discrete]): space of the node features.
            edge_space (Union[None, Box, Discrete]): space of the node features.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.
        """
        self._np_random = None
        if seed is not None:
            if isinstance(seed, seeding.RandomNumberGenerator):
                self._np_random = seed
            else:
                self.seed(seed)

        assert isinstance(
            node_space, (Box, Discrete)
        ), "Values of the node_space should be instances of Box or Discrete"
        if edge_space is not None:
            assert isinstance(
                edge_space, (Box, Discrete)
            ), "Values of the edge_space should be instances of Box or Discrete"

        self.node_space = node_space
        self.edge_space = edge_space

        # graph object creator
        self.graphObj = GraphObj

        super().__init__(None, None, seed)

    def _generate_sample_space(self, base_space, num) -> Optional[Union[Box, Discrete]]:
        # the possibility of this space having nothing
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
                "Only Box and Discrete can be accepted as a base_space."
            )

    def _sample_sample_space(self, sample_space) -> Optional[np.ndarray]:
        if sample_space is not None:
            return sample_space.sample()
        else:
            return None

    def sample(self) -> GraphObj:
        """Returns a random sized graph space with num_nodes between 1 and 10"""
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

        sampled_node_list = None
        if sampled_edges is not None:
            if num_edges > 0:
                sampled_node_list = self.np_random.integers(
                    low=0, high=num_edges, size=(num_edges, 2)
                )

        return GraphObj(sampled_nodes, sampled_edges, sampled_node_list)

    def contains(self, x: GraphObj) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, GraphObj):
            return False
        if x.nodes is not None:
            for node in x.nodes:
                if not self.node_space.contains(node):
                    return False
        if x.edges is not None:
            for edge in x.edges:
                if not self.edge_space.contains(edge):
                    return False

            if len(x.edge_links) != len(x.edges):
                return False

            if x.edge_links.shape[-1] != 2:
                return False

            if not np.issubdtype(x.edge_links.dtype, np.integer):
                return False

            if x.edge_links.max() >= len(x.edges):
                return False

            if x.edge_links.min() < 0:
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

    def to_jsonable(self, sample_n: GraphObj) -> list:
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
                ret_n = GraphObj(
                    np.asarray(sample["nodes"]),
                    np.asarray(sample["edges"]),
                    np.asarray(sample["edge_links"]),
                )
            else:
                ret_n = GraphObj(
                    np.asarray(sample["nodes"]),
                    None,
                    None,
                )
            ret.append(ret_n)
        return ret

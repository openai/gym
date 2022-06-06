from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.space import Space
from gym.utils import seeding


class GraphObj:
    def __init__(
        self,
        nodes: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        node_list: Optional[np.ndarray] = None,
    ):

        self.nodes = nodes
        self.edges = edges
        self.node_list = node_list

    def __repr__(self) -> str:
        return f"GraphObj(nodes: \n{self.nodes}, \n\nedges: \n{self.edges}, \n\nnode_list\n{self.node_list})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, GraphObj)
            and (self.nodes == other.nodes)
            and (self.edges == other.edges)
            and (self.node_list == other.node_list)
        )


class Graph(Space):
    """
    A dictionary representing graph spaces with `node_features`, `edge_features` and `edge_list`.

    Example usage::

        self.observation_space = spaces.Graph(node_space=space.Box(low=-100, high=100, shape=(3,)), edge_space=spaces.Discrete(3))
    """

    def __init__(
        self,
        node_space: Union[Box, Discrete],
        edge_space: Union[Box, Discrete],
        seed: Optional[int | seeding.RandomNumberGenerator] = None,
    ):
        self._np_random = None
        if seed is not None:
            if isinstance(seed, seeding.RandomNumberGenerator):
                self._np_random = seed
            else:
                self.seed(seed)

        assert isinstance(
            node_space, (Box, Discrete)
        ), "Values of the node_space should be instances of Box or Discrete"
        assert isinstance(
            edge_space, (Box, Discrete)
        ), "Values of the edge_space should be instances of Box or Discrete"

        self.node_space = node_space
        self.edge_space = edge_space

        # graph object creator
        self.graphObj = GraphObj

        super().__init__(None, None, seed)

    def _generate_sample_space(self, base_space, num) -> Optional[Box | Discrete]:
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
            return Discrete(
                n=base_space.n, seed=self._np_random, start=base_space.start
            )
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
        if num_edges > 0:
            sampled_node_list = self.np_random.integers(
                low=0, high=num_edges, size=(num_edges, 2)
            )

        return GraphObj(sampled_nodes, sampled_edges, sampled_node_list)

    def contains(self, x: GraphObj) -> bool:
        if not isinstance(x, GraphObj):
            return False
        if x.nodes is not None:
            for node in x.nodes:
                if not self.node_space.contains(node):
                    return False
        if x.edges is not None:
            for edge in x.edges:
                if not self.node_space.contains(edge):
                    return False

            if len(x.node_list) != len(x.edges):
                return False

            if x.node_list.shape[-1] != 2:
                return False

            if not np.issubdtype(x.node_list.dtype, np.integer):
                return False

            if x.node_list.max() >= len(x.edges):
                return False

            if x.node_list.min() < 0:
                return False

        return True

    def __repr__(self) -> str:
        return f"Graph({self.node_space}, {self.edge_space})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Graph)
            and (self.node_space == other.node_space)
            and (self.edge_space == other.edge_space)
        )

    def to_jsonable(self, sample_n: GraphObj) -> list[dict]:
        """Convert a batch of samples from this space to a JSONable data type."""
        # serialize as list of dicts
        ret_n = []
        for sample in sample_n:
            ret = {}
            ret["nodes"] = sample.nodes
            ret["edges"] = sample.edges
            ret["node_list"] = sample.node_list
            ret_n.append(ret)
        return ret_n

    def from_jsonable(self, sample_n: Sequence[dict]) -> list[GraphObj]:
        """Convert a JSONable data type to a batch of samples from this space."""
        ret = []
        for sample in sample_n:
            ret_n = GraphObj(
                sample_n["nodes"], sample_n["edges"], sample_n["node_list"]
            )
            ret.append(ret_n)
        return ret

from funcnodes import Shelf, NodeDecorator
from typing import Union, Optional, Callable, Literal
import numpy as np
from numpy.random import RandomState
from sklearn.base import ClusterMixin
from enum import Enum
from joblib import Memory

from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    FeatureAgglomeration,
    KMeans,
    BisectingKMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
    OPTICS,
    SpectralBiclustering,
    SpectralCoclustering,
)


class Affinity(Enum):
    EUCLIDEAN = "euclidean"
    PRECOMPUTED = "precomputed"

    @classmethod
    def default(cls):
        return cls.EUCLIDEAN.value


# @NodeDecorator(
#     node_id="affinity_propagation",
#     name="AffinityPropagation",
# )
def affinity_propagation(
    damping: float = 0.5,
    max_iter: int = 200,
    convergence_iter: int = 15,
    copy: bool = True,
    preference: Optional[Union[float, np.ndarray]] = None,
    affinity: Affinity = Affinity.default(),
    verbose: bool = False,
    random_state: Optional[Union[int, RandomState]] = None,
) -> ClusterMixin:
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, default=0.5
        Damping factor in the range `[0.5, 1.0)` is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).

    max_iter : int, default=200
        Maximum number of iterations.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy : bool, default=True
        Make a copy of input data.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : {'euclidean', 'precomputed'}, default='euclidean'
        Which affinity to use. At the moment 'precomputed' and
        ``euclidean`` are supported. 'euclidean' uses the
        negative squared euclidean distance between points.

    verbose : bool, default=False
        Whether to be verbose.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Attributes
    ----------
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Indices of cluster centers.

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AgglomerativeClustering : Recursively merges the pair of
        clusters that minimally increases a given linkage distance.
    FeatureAgglomeration : Similar to AgglomerativeClustering,
        but recursively merges features instead of samples.
    KMeans : K-Means clustering.
    MiniBatchKMeans : Mini-Batch K-Means clustering.
    MeanShift : Mean shift clustering using a flat kernel.
    SpectralClustering : Apply clustering to a projection
        of the normalized Laplacian.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    When the algorithm does not converge, it will still return a arrays of
    ``cluster_center_indices`` and labels if there are any exemplars/clusters,
    however they may be degenerate and should be used with caution.

    When ``fit`` does not converge, ``cluster_centers_`` is still populated
    however it may be degenerate. In such a case, proceed with caution.
    If ``fit`` does not converge and fails to produce any ``cluster_centers_``
    then ``predict`` will label every sample as ``-1``.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, ``fit`` will result in
    a single cluster center and label ``0`` for every sample. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> from sklearn.cluster import AffinityPropagation
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AffinityPropagation(random_state=5).fit(X)
    >>> clustering
    AffinityPropagation(random_state=5)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.predict([[0, 0], [4, 4]])
    array([0, 1])
    >>> clustering.cluster_centers_
    array([[1, 2],
           [4, 2]])
    Returns
    -------
    ClusterMixin: An instance of the AffinityPropagation class from scikit-learn.
    """

    def create_affinity_propagation():
        return AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            copy=copy,
            preference=preference,
            affinity=affinity,
            verbose=verbose,
            random_state=random_state,
        )

    return create_affinity_propagation()


class Metric(Enum):
    EUCLIDEAN = "euclidean"
    PRECOMPUTED = "precomputed"
    L1 = "l1"
    L2 = "l2"
    MANHATTAN = "manhattan"
    COSINE = "cosine"

    @classmethod
    def default(cls):
        return cls.EUCLIDEAN.value


class Linkage(Enum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"

    @classmethod
    def default(cls):
        return cls.WARD.value


# @NodeDecorator(
#     node_id="agglomerative_clustering",
#     name="AgglomerativeClustering",
# )
def agglomerative_clustering(
    n_clusters: int = 2,
    metric: Union[Metric, Callable] = Metric.default(),
    memory: Union[str, Memory] = None,
    connectivity: Optional[Union[np.ndarray, Callable]] = None,
    compute_full_tree: Union[Literal["auto"], bool] = "auto",
    linkage: Linkage = Linkage.default(),
    distance_threshold: Optional[float] = None,
    compute_distances: bool = False,
) -> ClusterMixin:
    """
    Agglomerative Clustering.

    Recursively merges pair of clusters of sample data; uses linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted. If "precomputed", a distance matrix is needed
        as input for the fit method.

        .. versionadded:: 1.2

        .. deprecated:: 1.4
           `metric=None` is deprecated in 1.4 and will be removed in 1.6.
           Let `metric` be the default value (i.e. `"euclidean"`) instead.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is ``None``, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - 'ward' minimizes the variance of the clusters being merged.
        - 'average' uses the average of the distances of each observation of
          the two sets.
        - 'complete' or 'maximum' linkage uses the maximum distances between
          all observations of the two sets.
        - 'single' uses the minimum of the distances between all observations
          of the two sets.

        .. versionadded:: 0.20
            Added the 'single' option

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    FeatureAgglomeration : Agglomerative clustering but for features instead of
        samples.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering
    AgglomerativeClustering()
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    Returns
    -------
    AgglomerativeClustering: An instance of the AgglomerativeClustering class from scikit-learn.
    """

    def create_agglomerative_clustering():
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
        )

    return create_agglomerative_clustering()


# @NodeDecorator(
#     node_id="birch",
#     name="Birch",
# )
def birch(
    threshold: float = 0.5,
    branching_factor: int = 50,
    n_clusters: Union[int, ClusterMixin, None] = 3,
    compute_labels: bool = True,
    copy: bool = True,
) -> ClusterMixin:
    """Implements the BIRCH clustering algorithm.

    It is a memory-efficient, online-learning algorithm provided as an
    alternative to :class:`MiniBatchKMeans`. It constructs a tree
    data structure with the cluster centroids being read off the leaf.
    These can be either the final cluster centroids or can be provided as input
    to another clustering algorithm such as :class:`AgglomerativeClustering`.

    Read more in the :ref:`User Guide <birch>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    threshold : float, default=0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default=50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    n_clusters : int, instance of sklearn.cluster model or None, default=3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples.

        - `None` : the final clustering step is not performed and the
          subclusters are returned as they are.

        - :mod:`sklearn.cluster` Estimator : If a model is provided, the model
          is fit treating the subclusters as new samples and the initial data
          is mapped to the label of the closest subcluster.

        - `int` : the model fit is :class:`AgglomerativeClustering` with
          `n_clusters` set to be equal to the int.

    compute_labels : bool, default=True
        Whether or not to compute labels for each fit.

    copy : bool, default=True
        Whether or not to make a copy of the given data. If set to False,
        the initial data will be overwritten.

    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.

    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    subcluster_labels_ : ndarray
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.

    labels_ : ndarray of shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative implementation that does incremental updates
        of the centers' positions using mini-batches.

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, squared sum and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, squared sum and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.

    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

    * Roberto Perdisci
      JBirch - Java implementation of BIRCH clustering algorithm
      https://code.google.com/archive/p/jbirch

    Examples
    --------
    >>> from sklearn.cluster import Birch
    >>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    >>> brc = Birch(n_clusters=None)
    >>> brc.fit(X)
    Birch(n_clusters=None)
    >>> brc.predict(X)
    array([0, 0, 0, 1, 1, 1])
    Returns
    -------
    Birch: An instance of the Birch class from scikit-learn.
    """

    def create_birch():
        return Birch(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
            compute_labels=compute_labels,
            copy=copy,
        )

    return create_birch()


# @NodeDecorator(
#     node_id="dbscan",
#     name="DBSCAN",
# )


class Algorithm(Enum):
    AUTO = "auto"
    BRUTE = "brute"
    KD_TREE = "kd_tree"
    BALL_TREE = "ball_tree"

    @classmethod
    def default(cls):
        return cls.AUTO.value


# @NodeDecorator(
#     node_id="dbscan",
#     name="DBSCAN",
# )
def dbscan(
    eps: float = 0.5,
    min_samples: int = 5,
    metric: Union[Metric, Callable] = Metric.default(),
    metric_params: Optional[dict] = None,
    algorithm: Algorithm = Algorithm.default(),
    leaf_size: int = 30,
    p: Optional[float] = None,
    n_jobs: Optional[int] = None,
) -> ClusterMixin:
    """Perform DBSCAN clustering from vector array or distance matrix.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    The worst case memory complexity of DBSCAN is :math:`O({n}^2)`, which can
    occur when the `eps` param is large and `min_samples` is low.

    Read more in the :ref:`User Guide <dbscan>`.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point. This includes the point itself. If
        `min_samples` is set to a higher value, DBSCAN will find denser clusters,
        whereas if it is set to a lower value, the found clusters will be more
        sparse.

    metric : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors for DBSCAN.

        .. versionadded:: 0.17
           metric *precomputed* to accept precomputed sparse matrix.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 0.19

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=None
        The power of the Minkowski metric to be used to calculate distance
        between points. If None, then ``p=2`` (equivalent to the Euclidean
        distance).

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.

    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    OPTICS : A similar clustering at multiple values of eps. Our implementation
        is optimized for memory usage.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_dbscan.py
    <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

    This implementation bulk-computes all neighborhood queries, which increases
    the memory complexity to O(n.d) where d is the average number of neighbors,
    while original DBSCAN had memory complexity O(n). It may attract a higher
    memory complexity when querying these nearest neighborhoods, depending
    on the ``algorithm``.

    One way to avoid the query complexity is to pre-compute sparse
    neighborhoods in chunks using
    :func:`NearestNeighbors.radius_neighbors_graph
    <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
    ``mode='distance'``, then using ``metric='precomputed'`` here.

    Another way to reduce memory and computation time is to remove
    (near-)duplicate points and use ``sample_weight`` instead.

    :class:`~sklearn.cluster.OPTICS` provides a similar clustering with lower memory
    usage.

    References
    ----------
    Ester, M., H. P. Kriegel, J. Sander, and X. Xu, `"A Density-Based
    Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
    <https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf>`_.
    In: Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996

    Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017).
    :doi:`"DBSCAN revisited, revisited: why and how you should (still) use DBSCAN."
    <10.1145/3068335>`
    ACM Transactions on Database Systems (TODS), 42(3), 19.

    Examples
    --------
    >>> from sklearn.cluster import DBSCAN
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 2], [2, 3],
    ...               [8, 7], [8, 8], [25, 80]])
    >>> clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    >>> clustering.labels_
    array([ 0,  0,  0,  1,  1, -1])
    >>> clustering
    DBSCAN(eps=3, min_samples=2)

    Returns
    -------
    DBSCAN: An instance of the DBSCAN class from scikit-learn.
    """

    def create_dbscan():
        return DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )

    return create_dbscan()


# @NodeDecorator(
#     node_id="feature_agglomeration",
#     name="FeatureAgglomeration",
# )
def feature_agglomeration(
    n_clusters: Union[int, None] = 2,
    metric: Union[Metric, Callable] = Metric.default(),
    memory: Union[str, Memory] = None,
    connectivity: Optional[Union[np.ndarray, Callable]] = None,
    compute_full_tree: Union[Literal["auto"], bool] = "auto",
    linkage: Linkage = Linkage.default(),
    pooling_func: Callable = np.mean,
    distance_threshold: Optional[float] = None,
    compute_distances: bool = False,
) -> ClusterMixin:
    """Agglomerate features.

    Recursively merges pair of clusters of features.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted. If "precomputed", a distance matrix is needed
        as input for the fit method.

        .. versionadded:: 1.2

        .. deprecated:: 1.4
           `metric=None` is deprecated in 1.4 and will be removed in 1.6.
           Let `metric` be the default value (i.e. `"euclidean"`) instead.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like or callable, default=None
        Connectivity matrix. Defines for each feature the neighboring
        features following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is `None`, i.e, the
        hierarchical clustering algorithm is unstructured.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at `n_clusters`. This is useful
        to decrease computation time if the number of clusters is not small
        compared to the number of features. This option is useful only when
        specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {"ward", "complete", "average", "single"}, default="ward"
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of features. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - "ward" minimizes the variance of the clusters being merged.
        - "complete" or maximum linkage uses the maximum distances between
          all features of the two sets.
        - "average" uses the average of the distances of each feature of
          the two sets.
        - "single" uses the minimum of the distances between all features
          of the two sets.

    pooling_func : callable, default=np.mean
        This combines the values of agglomerated features into a single
        value, and should accept an array of shape [M, N] and the keyword
        argument `axis=1`, and reduce it to an array of size [M].

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : array-like of (n_features,)
        Cluster labels for each feature.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_features`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_features` is a non-leaf
        node and has children `children_[i - n_features]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_features + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    AgglomerativeClustering : Agglomerative clustering samples instead of
        features.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets, cluster
    >>> digits = datasets.load_digits()
    >>> images = digits.images
    >>> X = np.reshape(images, (len(images), -1))
    >>> agglo = cluster.FeatureAgglomeration(n_clusters=32)
    >>> agglo.fit(X)
    FeatureAgglomeration(n_clusters=32)
    >>> X_reduced = agglo.transform(X)
    >>> X_reduced.shape
    (1797, 32)

    Returns
    -------
    FeatureAgglomeration: An instance of the FeatureAgglomeration class from scikit-learn.
    """

    def create_feature_agglomeration():
        return FeatureAgglomeration(
            n_clusters=n_clusters,
            metric=metric,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            pooling_func=pooling_func,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
        )

    return create_feature_agglomeration()



class KMeansAlgorithm(Enum):
    LLOYD = "lloyd"
    ELKAN = "elkan"
    
    @classmethod
    def default(cls):
        return cls.LLOYD.value
# @NodeDecorator(
#     node_id="kmeans",
#     name="KMeans",
# )
def kmeans(
    n_clusters: int = 8,
    init: Union[str, np.ndarray, Callable] = "k-means++",
    n_init: Union[Literal["auto"], int] = "auto",
    max_iter: int = 300,
    tol: float = 1e-4,
    verbose: int = 0,
    random_state: Optional[Union[int, RandomState]] = None,
    copy_x: bool = True,
    algorithm: KMeansAlgorithm = KMeansAlgorithm.default(),
) -> ClusterMixin:
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

        For an example of how to choose an optimal value for `n_clusters` refer to
        :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py`.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        * 'k-means++' : selects initial cluster centroids using sampling \
            based on an empirical probability distribution of the points' \
            contribution to the overall inertia. This technique speeds up \
            convergence. The algorithm implemented is "greedy k-means++". It \
            differs from the vanilla k-means++ by making several trials at \
            each sampling step and choosing the best centroid among them.

        * 'random': choose `n_clusters` observations (rows) at random from \
        data for the initial centroids.

        * If an array is passed, it should be of shape (n_clusters, n_features)\
        and gives the initial centers.

        * If a callable is passed, it should take arguments X, n_clusters and a\
        random state and return an initialization.

        For an example of how to use the different `init` strategy, see the example
        entitled :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`.

    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` changed to `'auto'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), where n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features.
    Refer to :doi:`"How slow is the k-means method?" D. Arthur and S. Vassilvitskii -
    SoCG2006.<10.1145/1137856.1137880>` for more details.

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])

    For a more detailed example of K-Means using the iris dataset see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_iris.py`.

    For examples of common problems with K-Means and how to address them see
    :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_assumptions.py`.

    For an example of how to use K-Means to perform color quantization see
    :ref:`sphx_glr_auto_examples_cluster_plot_color_quantization.py`.

    For a demonstration of how K-Means can be used to cluster text documents see
    :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`.

    For a comparison between K-Means and MiniBatchKMeans refer to example
    :ref:`sphx_glr_auto_examples_cluster_plot_mini_batch_kmeans.py`.
    
    Returns
    -------
    KMeans: An instance of the KMeans class from scikit-learn.
    """
    if isinstance(init, str) and init not in ["k-means++", "random"]:
        raise ValueError(
            "Invalid value for 'init': It must be np.ndarray, Callable or one of 'k-means++' or 'random'"
        )

    def create_kmeans():
        return KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )

    return create_kmeans()


class BisectingStrategy(Enum):
    BIGGEST_INERTIA = "biggest_inertia"
    LARGEST_CLUSTER = "largest_cluster"
    
    @classmethod
    def default(cls):
        return cls.BIGGEST_INERTIA.value

# @NodeDecorator(
#     node_id="bisecting_kmeans",
#     name="BisectingKMeans",
# )
def bisecting_kmeans(
    n_clusters: int = 8,
    init: Union[str, np.ndarray, Callable] = "k-means++",
    n_init: int = 1,
    random_state: Optional[Union[int, RandomState]] = None,
    max_iter: int = 100,
    verbose: int = 0,
    tol: float = 1e-4,
    copy_x: bool = True,
    algorithm: KMeansAlgorithm = KMeansAlgorithm.default(),    
    bisecting_strategy: BisectingStrategy = BisectingStrategy.default(),
) -> ClusterMixin:
    """Bisecting K-Means clustering.

    Read more in the :ref:`User Guide <bisect_k_means>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'} or callable, default='random'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=1
        Number of time the inner k-means algorithm will be run with different
        centroid seeds in each bisection.
        That will result producing for each bisection best output of n_init
        consecutive runs in terms of inertia.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization
        in inner K-Means. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    max_iter : int, default=300
        Maximum number of iterations of the inner k-means algorithm at each
        bisection.

    verbose : int, default=0
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations  to declare
        convergence. Used in inner k-means algorithm at each bisection to pick
        best possible clusters.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Inner K-means algorithm used in bisection.
        The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

    bisecting_strategy : {"biggest_inertia", "largest_cluster"},\
            default="biggest_inertia"
        Defines how bisection should be performed:

         - "biggest_inertia" means that BisectingKMeans will always check
            all calculated cluster for cluster with biggest SSE
            (Sum of squared errors) and bisect it. This approach concentrates on
            precision, but may be costly in terms of execution time (especially for
            larger amount of data points).

         - "largest_cluster" - BisectingKMeans will always split cluster with
            largest amount of points assigned to it from all clusters
            previously calculated. That should work faster than picking by SSE
            ('biggest_inertia') and may produce similar results in most cases.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    KMeans : Original implementation of K-Means algorithm.

    Notes
    -----
    It might be inefficient when n_cluster is less than 3, due to unnecessary
    calculations for that case.

    Examples
    --------
    >>> from sklearn.cluster import BisectingKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 1], [10, 1], [3, 1],
    ...               [10, 0], [2, 1], [10, 2],
    ...               [10, 8], [10, 9], [10, 10]])
    >>> bisect_means = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
    >>> bisect_means.labels_
    array([0, 2, 0, 2, 0, 2, 1, 1, 1], dtype=int32)
    >>> bisect_means.predict([[0, 0], [12, 3]])
    array([0, 2], dtype=int32)
    >>> bisect_means.cluster_centers_
    array([[ 2., 1.],
           [10., 9.],
           [10., 1.]])
    Returns
    -------
    BisectingKMeans: An instance of the BisectingKMeans class from scikit-learn.
    """
    def create_bisecting_kmeans():
        return BisectingKMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            copy_x=copy_x,
            algorithm=algorithm,
            n_init=n_init,
            bisecting_strategy=bisecting_strategy
        )

    return create_bisecting_kmeans()
# @NodeDecorator(
#     node_id="mini_batch_kmeans",
#     name="Mini Batch KMeans",
# )
# def mini_batch_kmeans(
#     n_clusters: int = 8,
#     init: Union[str, ndarray, 'callable'] = 'k-means++',
#     max_iter: int = 100,
#     batch_size: int = 100,
#     verbose: int = 0,
#     compute_labels: bool = True,
#     random_state: Optional[Union[int, RandomState]] = None,
#     tol: float = 0.0,
#     max_no_improvement: Optional[int] = None,
#     init_size: Optional[int] = None,
#     n_init: int = 3,
#     reassignment_ratio: float = 0.01,
# ) -> ClusterMixin:
#     """Mini Batch KMeans Clustering.

#     Read more in the :ref:`User Guide <mini_batch_kmeans>`.

#     Parameters
#     ----------

#     n_clusters : int, default=8
#         The number of clusters to form as well as the number of centroids
#         to generate.

#     init : {'k-means++', 'random', ndarray, callable}, default='k-means++'
#         Method for initialization:

#         'k-means++' : selects initial cluster centers for k-mean clustering
#         in a smart way to speed up convergence. See section Notes in
#         k_init for more details.

#         'random': choose n_clusters observations (rows) at random from data
#         for the initial centroids.

#         If an ndarray is passed, it should be of shape (n_clusters, n_features)
#         and gives the initial centers.

#         If a callable is passed, it should take arguments X, n_clusters and
#         a random state and return an initialization.

#     max_iter : int, default=100
#         Maximum number of iterations over the complete dataset before stopping.

#     batch_size : int, default=100
#         Size of the mini batches.

#     verbose : int, default=0
#         Verbosity mode.

#     compute_labels : bool, default=True
#         Compute label assignment and inertia for the complete dataset once the
#         minibatch optimization has converged in fit. By default, this is
#         activated. It may be deactivated if the user wants to manually extract
#         the labels through the labels_ attribute after calling fit.

#     random_state : int, RandomState instance or None, default=None
#         Determines random number generation for centroid initialization. Use
#         an int to make the randomness deterministic.
#         See :term:`Glossary <random_state>`.

#     tol : float, default=0.0
#         Control early stopping based on the relative center changes as measured
#         by the inertia metric. If the relative change is below tol, the model
#         is considered to have converged and the optimization stops.

#     max_no_improvement : int, default=None
#         Control early stopping based on the consecutive number of mini batches
#         that does not yield an improvement on the smoothed inertia. To disable
#         convergence detection based on inertia, set max_no_improvement to None.

#     init_size : int, default=None
#         Number of samples to randomly sample for speeding up the initialization.
#         Deactivated if set to None.

#     n_init : int, default=3
#         Number of random initializations that are tried.

#     reassignment_ratio : float, default=0.01
#         Control threshold for early stopping in the computation of the mini
#         batch KMeans algorithm. When a minibatch is deemed to have converged
#         (i.e., the maximum number of iterations, max_iter, has been reached
#         or the no_improvement consecutive batches have been processed), it
#         is assigned a label and center update may happen. If the fraction of
#         the current batch that is reassigned to a different center is lower
#         than this threshold, the algorithm does not actually reassign the
#         centers of the centers and the reassigned ratio is instead shown in
#         the early stopping message, followed by the number of samples being
#         reclustered.

#     Returns
#     -------
#     MiniBatchKMeans: An instance of the MiniBatchKMeans class from scikit-learn.
#     """
#     def create_mini_batch_kmeans():
#         return MiniBatchKMeans(
#             n_clusters=n_clusters,
#             init=init,
#             max_iter=max_iter,
#             batch_size=batch_size,
#             verbose=verbose,
#             compute_labels=compute_labels,
#             random_state=random_state,
#             tol=tol,
#             max_no_improvement=max_no_improvement,
#             init_size=init_size,
#             n_init=n_init,
#             reassignment_ratio=reassignment_ratio,
#         )

#     return create_mini_batch_kmeans
# @NodeDecorator(
#     node_id="mean_shift",
#     name="Mean Shift",
# )
# def mean_shift(
#     bandwidth: Optional[float] = None,
#     seeds: Optional[array_like] = None,
#     bin_seeding: bool = False,
#     min_bin_freq: int = 1,
#     cluster_all: bool = True,
#     n_jobs: Optional[int] = None,
#     max_iter: int = 300,
#     verbose: int = 0,
# ) -> ClusterMixin:
#     """Mean Shift Clustering.

#     Read more in the :ref:`User Guide <mean_shift>`.

#     Parameters
#     ----------

#     bandwidth : float, optional
#         Bandwidth used in the RBF kernel. If not given, the bandwidth is
#         estimated using sklearn.cluster.estimate_bandwidth.

#     seeds : array-like of shape (n_samples, n_features), default=None
#         Seeds used to initialize kernels. If None and bin_seeding=False,
#         seeds are calculated by clustering.get_bin_seeds with min_bin_freq.

#     bin_seeding : bool, default=False
#         If true, initial kernel locations are not locations of all
#         points, but rather the location of the discretized version of
#         points, where points are binned onto a grid whose coarseness
#         corresponds to the bandwidth.

#     min_bin_freq : int, default=1
#         To speed up the algorithm, accept only those bins with
#         at least min_bin_freq points as seeds.

#     cluster_all : bool, default=True
#         If true, then all points are clustered, even those orphans
#         with no nearby seeds. Orphans are assigned to the nearest
#         kernel. If False, then all orphans are left unclustered.

#     n_jobs : int or None, default=None
#         The number of jobs to use for the computation. This works by
#         computing each of the n_init runs in parallel.

#     max_iter : int, default=300
#         Maximum number of iterations of the mean shift algorithm for a
#         single run.

#     verbose : int, default=0
#         Verbosity mode.

#     Returns
#     -------
#     MeanShift: An instance of the MeanShift class from scikit-learn.
#     """
#     def create_mean_shift():
#         return MeanShift(
#             bandwidth=bandwidth,
#             seeds=seeds,
#             bin_seeding=bin_seeding,
#             min_bin_freq=min_bin_freq,
#             cluster_all=cluster_all,
#             n_jobs=n_jobs,
#             max_iter=max_iter,
#             verbose=verbose,
#         )

#     return create_mean_shift
# @NodeDecorator(
#     node_id="optics",
#     name="OPTICS",
# )
# def optics(
#     min_samples: int = 5,
#     max_eps: float = np.inf,
#     metric: str = 'minkowski',
#     p: int = 2,
#     metric_params: Optional[dict] = None,
#     cluster_method: str = 'xi',
#     eps: Optional[float] = None,
#     xi: float = 0.05,
#     predecessor_correction: bool = True,
#     min_cluster_size: Optional[int] = None,
#     algorithm: str = 'auto',
#     leaf_size: int = 30,
#     n_jobs: Optional[int] = None,
# ) -> ClusterMixin:
#     """OPTICS Clustering.

#     Read more in the :ref:`User Guide <optics>`.

#     Parameters
#     ----------

#     min_samples : int, default=5
#         The number of samples in a neighborhood for a point to be considered
#         as a core point. This includes the point itself.

#     max_eps : float, default=np.inf
#         The maximum distance between two samples for one to be considered
#         as in the neighborhood of the other. This is used when constructing
#         the reachability matrix, which can be thought of as the minimum
#         spanning tree of the core samples.

#     metric : str or callable, default='minkowski'
#         The metric to use when calculating distance between instances in a
#         feature array. If metric is a string or callable, it must be one of
#         the options allowed by metrics.pairwise.calculate_distance for its
#         metric parameter. If metric is "precomputed", X is assumed to be a
#         distance matrix and must be square during fit. X may be also provided
#         as a sparse matrix. If metric is a callable function, it is called on
#         each pair of instances (rows) and the resulting value recorded. The
#         callable should take two arrays from X as input and return a value
#         indicating the distance between them. This works for Scipy's metrics,
#         but is less efficient than passing the metric name as a string.

#     p : int, default=2
#         The power of the Minkowski metric to be used to calculate distance
#         between points.

#     metric_params : dict, default=None
#         Additional keyword arguments for the metric function.

#     cluster_method : {'xi', 'dbscan'}, default='xi'
#         The extraction method used to extract clusters. 'xi' uses the
#         Xi method of cluster extraction, while 'dbscan' uses the DBSCAN
#         algorithm to extract clusters.

#     eps : float, default=None
#         The maximum distance between two samples for one to be considered
#         as in the neighborhood of the other. This is used when constructing
#         the reachability matrix, which can be thought of as the minimum
#         spanning tree of the core samples. Ignored if cluster_method='dbscan'.

#     xi : float, default=0.05
#         Determines the minimum steepness of the reachability plot for a point
#         to be considered a local maximum. Only used if cluster_method='xi'.

#     predecessor_correction : bool, default=True
#         Perform predecessor correction to make sure cluster order is maintained
#         in hierarchical clustering. This can also improve clustering performance
#         for large datasets.

#     min_cluster_size : int or None, default=None
#         Minimum number of samples in an OPTICS cluster, expressed as an absolute
#         number or a fraction of the number of samples (performs better with
#         larger value). If None, use min_samples.

#     algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
#         Algorithm to use for nearest neighbors search. 'auto' will attempt
#         to decide the most appropriate algorithm based on the values passed
#         to fit method.

#     leaf_size : int, default=30
#         Leaf size passed to BallTree or KDTree. This can affect the speed
#         of the construction and query, as well as the memory required to
#         store the tree. The optimal value depends on the nature of the
#         problem.

#     n_jobs : int or None, default=None
#         The number of jobs to use for the computation. This works by computing
#         each of the n_init runs in parallel.

#     Returns
#     -------
#     OPTICS: An instance of the OPTICS class from scikit-learn.
#     """
#     def create_optics():
#         return OPTICS(
#             min_samples=min_samples,
#             max_eps=max_eps,
#             metric=metric,
#             p=p,
#             metric_params=metric_params,
#             cluster_method=cluster_method,
#             eps=eps,
#             xi=xi,
#             predecessor_correction=predecessor_correction,
#             min_cluster_size=min_cluster_size,
#             algorithm=algorithm,
#             leaf_size=leaf_size,
#             n_jobs=n_jobs,
#         )

#     return create_optics
# @NodeDecorator(
#     node_id="spectral_clustering",
#     name="Spectral Clustering",
# )
# def spectral_clustering(
#     n_clusters: int = 8,
#     eigen_solver: Optional[str] = None,
#     random_state: Optional[Union[int, RandomState]] = None,
#     n_init: int = 10,
#     gamma: float = 1.0,
#     affinity: Union[str, Callable] = 'rbf',
#     n_neighbors: int = 10,
#     eigen_tol: float = 0.0,
#     assign_labels: str = 'kmeans',
#     degree: int = 3,
#     coef0: float = 1,
#     kernel_params: Optional[dict] = None,
#     n_jobs: Optional[int] = None,
# ) -> ClusterMixin:
#     """Spectral Clustering.

#     Read more in the :ref:`User Guide <spectral_clustering>`.

#     Parameters
#     ----------

#     n_clusters : int, default=8
#         The number of clusters to form as well as the number of centroids
#         to generate.

#     eigen_solver : {'arpack', 'lobpcg', None}, default=None
#         The eigenvalue decomposition strategy to use. AMG requires pyamg
#         to be installed. It can be faster on very large, sparse problems,
#         but may also lead to instabilities. Lobpcg can be used if arpack
#         crashes.

#     random_state : int, RandomState instance or None, default=None
#         Determines random number generation for eigenvectors decomposition.
#         Use an int to make the randomness deterministic.
#         See :term:`Glossary <random_state>`.

#     n_init : int, default=10
#         Number of time the k-means algorithm will be run with different
#         centroid seeds. The final results will be the best output of
#         n_init consecutive runs in terms of inertia.

#     gamma : float, default=1.0
#         Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
#         Ignored for affinity='nearest_neighbors' and affinity='precomputed'.

#     affinity : str or callable, default='rbf'
#         How to construct the affinity matrix.

#         - 'nearest_neighbors' : construct the affinity matrix by computing
#           a graph of nearest neighbors.
#         - 'rbf' : construct the affinity matrix using a radial basis function
#           (RBF) kernel.
#         - 'precomputed' : interpret X as a precomputed affinity matrix.
#         - callable : use passed in function as affinity.

#     n_neighbors : int, default=10
#         Number of neighbors to use when constructing the affinity matrix
#         using the nearest neighbors method. Ignored for affinity='rbf'.

#     eigen_tol : float, default=0.0
#         Stopping criterion for eigendecomposition of the Laplacian matrix
#         when using arpack eigen_solver.

#     assign_labels : {'kmeans', 'discretize'}, default='kmeans'
#         The strategy to use to assign labels in the embedding space.
#         There are two ways to assign labels after the laplacian embedding.
#         k-means can be applied and is a popular choice. But it can also be
#         sensitive to initialization. Discretization is another approach which
#         is less sensitive to random initialization.

#     degree : int, default=3
#         Degree of the polynomial kernel. Ignored by other kernels.

#     coef0 : float, default=1
#         Zero coefficient for polynomial and sigmoid kernels.
#         Ignored by other kernels.

#     kernel_params : dict of str to any, default=None
#         Parameters (keyword arguments) and values for kernel passed as
#         callable object. Ignored by other kernels.

#     n_jobs : int or None, default=None
#         The number of parallel jobs to run for neighbor search and eigendecomposition.
#         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#         ``-1`` means using all processors.

#     Returns
#     -------
#     SpectralClustering: An instance of the SpectralClustering class from scikit-learn.
#     """
#     def create_spectral_clustering():
#         return SpectralClustering(
#             n_clusters=n_clusters,
#             eigen_solver=eigen_solver,
#             random_state=random_state,
#             n_init=n_init,
#             gamma=gamma,
#             affinity=affinity,
#             n_neighbors=n_neighbors,
#             eigen_tol=eigen_tol,
#             assign_labels=assign_labels,
#             degree=degree,
#             coef0=coef0,
#             kernel_params=kernel_params,
#             n_jobs=n_jobs,
#         )

#     return create_spectral_clustering
# @NodeDecorator(
#     node_id="spectral_biclustering",
#     name="Spectral Biclustering",
# )
# def spectral_biclustering(
#     n_clusters: int = 3,
#     method: str = 'bistochastic',
#     n_components: Optional[int] = None,
#     random_state: Optional[Union[int, RandomState]] = None,
#     n_init: int = 10,
#     svd_method: str = 'randomized',
#     mini_batch: bool = False,
#     n_jobs: Optional[int] = None,
# ) -> ClusterMixin:
#     """Spectral Biclustering.

#     Read more in the :ref:`User Guide <spectral_biclustering>`.

#     Parameters
#     ----------

#     n_clusters : int, default=3
#         The number of biclusters to find.

#     method : {'scale', 'bistochastic'}, default='bistochastic'
#         Method for the initialization of the spectral decomposition. Currently,
#         'scale' and 'bistochastic' are supported. 'scale' is the standard scaling
#         approach that ensures row-stochastic matrices. 'bistochastic' ensures
#         doubly-stochastic matrices.

#     n_components : int or None, default=None
#         Number of singular vectors to use for the spectral initialization. If None,
#         it is automatically chosen to be n_clusters.

#     random_state : int, RandomState instance or None, default=None
#         Determines random number generation for initialization. Use an int to make
#         the randomness deterministic.
#         See :term:`Glossary <random_state>`.

#     n_init : int, default=10
#         Number of times the algorithm will be run with different initializations.
#         The final results will be the best output of n_init consecutive runs in
#         terms of inertia.

#     svd_method : str, default='randomized'
#         SVD solver to use. If 'randomized', use a randomized algorithm. If 'arpack',
#         use the ARPACK wrapper in SciPy. For most problems, 'randomized' will
#         be faster.

#     mini_batch : bool, default=False
#         Whether to use a minibatch version of the SVD solver. The minibatch
#         SVD is computed on smaller, random subsets of the input data, which
#         can result in faster computation at the cost of accuracy.

#     n_jobs : int or None, default=None
#         The number of parallel jobs to run for SVD computation. ``None`` means
#         1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using
#         all processors.

#     Returns
#     -------
#     SpectralBiclustering: An instance of the SpectralBiclustering class from scikit-learn.
#     """
#     def create_spectral_biclustering():
#         return SpectralBiclustering(
#             n_clusters=n_clusters,
#             method=method,
#             n_components=n_components,
#             random_state=random_state,
#             n_init=n_init,
#             svd_method=svd_method,
#             mini_batch=mini_batch,
#             n_jobs=n_jobs,
#         )

#     return create_spectral_biclustering
# @NodeDecorator(
#     node_id="spectral_coclustering",
#     name="Spectral Co-clustering",
# )
# def spectral_coclustering(
#     n_clusters: int = 3,
#     svd_method: str = 'randomized',
#     n_init: int = 10,
#     random_state: Optional[Union[int, RandomState]] = None,
#     n_jobs: Optional[int] = None,
# ) -> ClusterMixin:
#     """Spectral Co-clustering.

#     Read more in the :ref:`User Guide <spectral_coclustering>`.

#     Parameters
#     ----------

#     n_clusters : int, default=3
#         The number of co-clusters to find.

#     svd_method : str, default='randomized'
#         SVD solver to use. If 'randomized', use a randomized algorithm.
#         If 'arpack', use the ARPACK wrapper in SciPy. For most problems,
#         'randomized' will be faster.

#     n_init : int, default=10
#         Number of times the algorithm will be run with different initializations.
#         The final results will be the best output of n_init consecutive runs
#         in terms of inertia.

#     random_state : int, RandomState instance or None, default=None
#         Determines random number generation for initialization. Use an int to
#         make the randomness deterministic.
#         See :term:`Glossary <random_state>`.

#     n_jobs : int or None, default=None
#         The number of parallel jobs to run for SVD computation. ``None`` means
#         1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using
#         all processors.

#     Returns
#     -------
#     SpectralCoclustering: An instance of the SpectralCoclustering class from scikit-learn.
#     """
#     def create_spectral_coclustering():
#         return SpectralCoclustering(
#             n_clusters=n_clusters,
#             svd_method=svd_method,
#             n_init=n_init,
#             random_state=random_state,
#             n_jobs=n_jobs,
#         )

#     return create_spectral_coclustering

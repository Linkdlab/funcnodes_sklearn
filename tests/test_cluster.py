import numpy as np
import unittest
from unittest.mock import patch, Mock
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    FeatureAgglomeration,
    KMeans,
)
from funcnodes_sklearn.cluster import (
    affinity_propagation,
    Affinity,
    agglomerative_clustering,
    Metric,
    Linkage,
    birch,
    dbscan,
    Algorithm,
    kmeans,
    KMeansAlgorithm,
    feature_agglomeration,
)
from joblib import Memory


class TestAffinityPropagation(unittest.TestCase):
    def test_default_parameters(self):
        clustering = affinity_propagation()
        self.assertIsInstance(clustering, AffinityPropagation)
        self.assertEqual(clustering.damping, 0.5)
        self.assertEqual(clustering.max_iter, 200)
        self.assertEqual(clustering.convergence_iter, 15)
        self.assertTrue(clustering.copy)
        self.assertIsNone(clustering.preference)
        self.assertEqual(clustering.affinity, Affinity.default())
        self.assertFalse(clustering.verbose)
        self.assertIsNone(clustering.random_state)

    def test_custom_parameters(self):
        damping = 0.7
        max_iter = 300
        convergence_iter = 20
        copy = False
        preference = np.array([0.1, 0.2, 0.3])
        affinity = Affinity.PRECOMPUTED.value
        verbose = True
        random_state = 42

        clustering = affinity_propagation(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            copy=copy,
            preference=preference,
            affinity=affinity,
            verbose=verbose,
            random_state=random_state,
        )

        self.assertIsInstance(clustering, AffinityPropagation)
        self.assertEqual(clustering.damping, damping)
        self.assertEqual(clustering.max_iter, max_iter)
        self.assertEqual(clustering.convergence_iter, convergence_iter)
        self.assertFalse(clustering.copy)
        np.testing.assert_array_equal(clustering.preference, preference)
        self.assertEqual(clustering.affinity, Affinity.PRECOMPUTED.value)
        self.assertTrue(clustering.verbose)
        self.assertEqual(clustering.random_state, random_state)

    # @patch('funcnodes_sklearn.cluster.affinity_propagation')
    # def test_creation_function(self, mock_affinity_propagation):
    #     clustering = affinity_propagation()
    #     mock_affinity_propagation.assert_called_once_with(
    #         damping=0.5,
    #         max_iter=200,
    #         convergence_iter=15,
    #         copy=True,
    #         preference=None,
    #         affinity=Affinity.EUCLIDEAN.value,
    #         verbose=False,
    #         random_state=None
    #     )


class TestAgglomerativeClustering(unittest.TestCase):
    def test_default_parameters(self):
        clustering = agglomerative_clustering()
        self.assertIsInstance(clustering, AgglomerativeClustering)
        self.assertEqual(clustering.n_clusters, 2)
        self.assertEqual(clustering.metric, Metric.default())
        self.assertIsNone(clustering.memory)
        self.assertIsNone(clustering.connectivity)
        self.assertEqual(clustering.compute_full_tree, "auto")
        self.assertEqual(clustering.linkage, Linkage.default())
        self.assertIsNone(clustering.distance_threshold)
        self.assertFalse(clustering.compute_distances)

    def test_custom_parameters(self):
        n_clusters = 3
        metric = Metric.L1.value
        memory = "memory_cache"
        connectivity = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
        compute_full_tree = True
        linkage = Linkage.AVERAGE.value
        distance_threshold = 0.5
        compute_distances = True

        clustering = agglomerative_clustering(
            n_clusters=n_clusters,
            metric=metric,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
        )

        self.assertIsInstance(clustering, AgglomerativeClustering)
        self.assertEqual(clustering.n_clusters, n_clusters)
        self.assertEqual(clustering.metric, metric)
        self.assertEqual(clustering.memory, memory)
        self.assertIs(clustering.connectivity, connectivity)
        self.assertEqual(clustering.compute_full_tree, compute_full_tree)
        self.assertEqual(clustering.linkage, linkage)
        self.assertEqual(clustering.distance_threshold, distance_threshold)
        self.assertTrue(clustering.compute_distances)

    def test_memory_caching(self):
        memory = Memory(location="cachedir", verbose=0)
        clustering = agglomerative_clustering(memory=memory)
        self.assertIsInstance(clustering, AgglomerativeClustering)
        self.assertEqual(clustering.memory, memory)

    def test_callable_metric(self):
        def custom_metric(x, y):
            return np.sum(np.abs(x - y))

        clustering = agglomerative_clustering(metric=custom_metric)
        self.assertIsInstance(clustering, AgglomerativeClustering)
        self.assertEqual(clustering.metric, custom_metric)

    # def test_create_agglomerative_clustering(self):
    #     agglomerative_clustering.create_agglomerative_clustering = Mock(
    #         return_value=AgglomerativeClustering()
    #     )
    #     clustering = agglomerative_clustering()
    #     agglomerative_clustering.create_agglomerative_clustering.assert_called_once()
    #     self.assertIsInstance(clustering, AgglomerativeClustering)


class TestBirchFunction(unittest.TestCase):
    def test_default_parameters(self):
        cluster = birch()
        self.assertIsInstance(cluster, Birch)
        self.assertEqual(cluster.threshold, 0.5)
        self.assertEqual(cluster.branching_factor, 50)
        self.assertEqual(cluster.n_clusters, 3)
        self.assertTrue(cluster.compute_labels)
        self.assertTrue(cluster.copy)

    def test_custom_parameters(self):
        threshold = 0.2
        branching_factor = 30
        n_clusters = 5
        compute_labels = False
        copy = False

        cluster = birch(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
            compute_labels=compute_labels,
            copy=copy,
        )

        self.assertIsInstance(cluster, Birch)
        self.assertEqual(cluster.threshold, threshold)
        self.assertEqual(cluster.branching_factor, branching_factor)
        self.assertEqual(cluster.n_clusters, n_clusters)
        self.assertFalse(cluster.compute_labels)
        self.assertFalse(cluster.copy)

    def test_n_clusters_sklearn_model(self):
        n_clusters_model = AgglomerativeClustering(n_clusters=2)
        cluster = birch(n_clusters=n_clusters_model)
        self.assertIsInstance(cluster, Birch)
        self.assertEqual(cluster.n_clusters, n_clusters_model)

    def test_n_clusters_none(self):
        n_clusters = None
        cluster = birch(n_clusters=n_clusters)
        self.assertIsInstance(cluster, Birch)
        self.assertEqual(cluster.n_clusters, n_clusters)

    # def test_create_birch_function(self):
    #     mock_birch = Mock(spec=Birch)
    #     create_birch_mock = Mock(return_value=mock_birch)

    #     with unittest.mock.patch('your_module.Birch', create_birch_mock):
    #         birch_instance = birch()

    #     create_birch_mock.assert_called_once_with(
    #         threshold=0.5,
    #         branching_factor=50,
    #         n_clusters=3,
    #         compute_labels=True,
    #         copy=True
    #     )
    #     self.assertEqual(birch_instance, mock_birch)


class TestDBSCAN(unittest.TestCase):
    def test_default_parameters(self):
        clustering = dbscan()
        self.assertIsInstance(clustering, DBSCAN)
        self.assertEqual(clustering.eps, 0.5)
        self.assertEqual(clustering.min_samples, 5)
        self.assertEqual(clustering.metric, Metric.default())
        self.assertEqual(clustering.algorithm, Algorithm.default())
        self.assertEqual(clustering.leaf_size, 30)
        self.assertIsNone(clustering.p)
        self.assertIsNone(clustering.n_jobs)

    def test_custom_parameters(self):
        eps = 1.0
        min_samples = 10
        metric = Metric.MANHATTAN.value
        metric_params = {"p": 2}
        algorithm = Algorithm.KD_TREE.value
        leaf_size = 50
        p = 2
        n_jobs = -1

        clustering = dbscan(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )

        self.assertIsInstance(clustering, DBSCAN)
        self.assertEqual(clustering.eps, eps)
        self.assertEqual(clustering.min_samples, min_samples)
        self.assertEqual(clustering.metric, metric)
        self.assertEqual(clustering.algorithm, algorithm)
        self.assertEqual(clustering.leaf_size, leaf_size)
        self.assertEqual(clustering.p, p)
        self.assertEqual(clustering.n_jobs, n_jobs)

    # @patch('your_module.DBSCAN')
    # def test_create_dbscan(self, mock_dbscan):
    #     eps = 0.5
    #     min_samples = 5
    #     metric = 'euclidean'
    #     metric_params = None
    #     algorithm = Algorithm.AUTO
    #     leaf_size = 30
    #     p = None
    #     n_jobs = None

    #     dbscan_instance = MagicMock()
    #     mock_dbscan.return_value = dbscan_instance

    #     clustering = dbscan(
    #         eps=eps,
    #         min_samples=min_samples,
    #         metric=metric,
    #         metric_params=metric_params,
    #         algorithm=algorithm,
    #         leaf_size=leaf_size,
    #         p=p,
    #         n_jobs=n_jobs
    #     )

    #     mock_dbscan.assert_called_once_with(
    #         eps=eps,
    #         min_samples=min_samples,
    #         metric=metric,
    #         metric_params=metric_params,
    #         algorithm=algorithm.value,
    #         leaf_size=leaf_size,
    #         p=p,
    #         n_jobs=n_jobs
    #     )
    #     self.assertIs(clustering, dbscan_instance)


class TestFeatureAgglomeration(unittest.TestCase):
    def test_feature_agglomeration_default_values(self):
        agglomeration = feature_agglomeration()
        self.assertIsInstance(agglomeration, FeatureAgglomeration)
        self.assertEqual(agglomeration.n_clusters, 2)
        self.assertEqual(agglomeration.metric, Metric.default())
        self.assertIsNone(agglomeration.memory)
        self.assertIsNone(agglomeration.connectivity)
        self.assertEqual(agglomeration.compute_full_tree, "auto")
        self.assertEqual(agglomeration.linkage, Linkage.default())
        self.assertEqual(agglomeration.pooling_func, np.mean)
        self.assertIsNone(agglomeration.distance_threshold)
        self.assertFalse(agglomeration.compute_distances)

    def test_feature_agglomeration_custom_values(self):
        n_clusters = 3
        metric = Metric.L1.value
        memory = "memory_cache"
        connectivity = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
        compute_full_tree = True
        linkage = Linkage.AVERAGE.value
        pooling_func = (np.median,)
        distance_threshold = 0.5
        compute_distances = True

        clustering = feature_agglomeration(
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

        self.assertIsInstance(clustering, FeatureAgglomeration)
        self.assertEqual(clustering.n_clusters, n_clusters)
        self.assertEqual(clustering.metric, metric)
        self.assertEqual(clustering.memory, memory)
        self.assertIs(clustering.connectivity, connectivity)
        self.assertEqual(clustering.compute_full_tree, compute_full_tree)
        self.assertEqual(clustering.linkage, linkage)
        self.assertEqual(clustering.distance_threshold, distance_threshold)
        self.assertTrue(clustering.compute_distances)
        self.assertEqual(clustering.pooling_func, pooling_func)

    def test_memory_caching(self):
        memory = Memory(location="cachedir", verbose=0)
        clustering = feature_agglomeration(memory=memory)
        self.assertIsInstance(clustering, FeatureAgglomeration)
        self.assertEqual(clustering.memory, memory)

    #     mock_feature_agglomeration = Mock(return_value=FeatureAgglomeration())
    #     with unittest.mock.patch('funcnodes_sklearn.cluster.FeatureAgglomeration', mock_feature_agglomeration):
    #         agglomeration = feature_agglomeration(
    #             n_clusters=5,
    #             metric=Metric.L1.value,
    #             memory="some_memory",
    #             connectivity=[[0, 1], [1, 0]],
    #             compute_full_tree=True,
    #             linkage=Linkage.AVERAGE.value,
    #             pooling_func=np.median,
    #             distance_threshold=0.5,
    #             compute_distances=True,
    #         )
    #     mock_feature_agglomeration.assert_called_once_with(
    #         n_clusters=5,
    #         metric=Metric.L1.value,
    #         memory="some_memory",
    #         connectivity=[[0, 1], [1, 0]],
    #         compute_full_tree=True,
    #         linkage=Linkage.AVERAGE.value,
    #         pooling_func=np.median,
    #         distance_threshold=0.5,
    #         compute_distances=True,
    #     )
    #     self.assertIsInstance(agglomeration, FeatureAgglomeration)


class TestKMeans(unittest.TestCase):
    def test_kmeans_default(self):
        result = kmeans()
        self.assertIsInstance(result, KMeans)
        self.assertEqual(result.n_clusters, 8)
        self.assertEqual(result.init, "k-means++")
        self.assertEqual(result.n_init, "auto")
        self.assertEqual(result.max_iter, 300)
        self.assertEqual(result.tol, 1e-4)
        self.assertEqual(result.verbose, 0)
        self.assertIsNone(result.random_state)
        self.assertTrue(result.copy_x)
        self.assertEqual(result.algorithm, KMeansAlgorithm.default())

    def test_kmeans_custom(self):
        result = kmeans(
            n_clusters=5,
            init=np.array([[1, 2], [3, 4], [5, 6]]),
            n_init=3,
            max_iter=150,
            tol=1e-3,
            verbose=1,
            random_state=42,
            copy_x=False,
            algorithm=KMeansAlgorithm.ELKAN.value,
        )
        self.assertIsInstance(result, KMeans)
        self.assertEqual(result.n_clusters, 5)
        np.testing.assert_array_equal(result.init, np.array([[1, 2], [3, 4], [5, 6]]))
        self.assertEqual(result.n_init, 3)
        self.assertEqual(result.max_iter, 150)
        self.assertEqual(result.tol, 1e-3)
        self.assertEqual(result.verbose, 1)
        self.assertEqual(result.random_state, 42)
        self.assertFalse(result.copy_x)
        self.assertEqual(result.algorithm, KMeansAlgorithm.ELKAN.value)

    def test_kmeans_custom_init_callable(self):
        def custom_init(X, n_clusters):
            # Custom initialization logic here
            return np.random.rand(n_clusters, X.shape[1])

        result = kmeans(init=custom_init)
        self.assertIsInstance(result, KMeans)

    # def test_kmeans_invalid_init(self):
    #     with self.assertRaises(ValueError):
    #         kmeans(init='invalid')

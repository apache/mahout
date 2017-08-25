---
layout: algorithm
title: DBSCAN
theme:
   name: retro-mahout
---

### About

[DBSCAN Clustering](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
 is a simple and efficient density based clustering algorithm for group objects
 into clusters. DBSCAN differs from other clustering algorithms in the way it
 clusters objects. DBSCAN uses the density information of points to define clusters
 Each object is represented as a point in the multidimensional feature space. The algorithm
 uses the notion of core points, border points and noise points to perform the clustering.

 Parameters of the Algorithm: (Epsilon, Min points)
 * Epsilon (Eps) - denotes the radius of the sphere that defines the neighbourhood of a given data point
 * Min points (Minpts) - the minimum threshold for the number of points in a point p's Eps-neighbourhood to
 make p a core point
 
 Basic Definitions
 * Eps-neighbourhood of a point - The Eps-neighbourhood of a point 'p' is defined as the set of points S
   such that for each x 'belongs to' S, distance(p,x) <= eps
 * Core point - A point 'p' is called a core point if the number of points in its Eps-neighbourhood is >= Minpts
 * Border point - A point that is not a core point but such that there exists a 'p' in its neighbourhood
   such that p is a core point
 * Noise point - Any point that does not classify as a core point or border point is a noise point. Noise points
   do not belong to any cluster.

 Algorithm:
 The DBSCAN algorithm takes as parameters Eps and MinPts. Eps restricts the neighbourhood of a point and MinPts
 denotes the threshold for the number of neighbours in Eps-neighbours of point p, for it to form a cluster.
 The algorithm randomly chooses a point q, and finds it Eps-neighbourhood. If the number of points in the
 Eps-neighbourhood of q is less than MinPts, it is marked as a noise point. Otherwise it is marked as a core point
 and a new cluster is created. If p is a core point, iteratively a Eps-neighbourhood query is performed on each of
 its neighbours and points are added to the created cluster.  If no unvisited points can be added to cluster, the new
 cluster is complete and no points will be added to the cluster in subsequent iterations. Then another unvisited point q'
 is picked up and the same process is repeated. The algorithm terminates when all the points have been visited
 i.e. when some of the points are added to clusters and some are marked as noise points. The total number of
 Eps-neighbourhood queries performed is equal to the size of the data (n) and if no indexing data structure is
 used then the calculation of Eps-neighbourhood query involved the distance calculation with all the points (n).
 Thus, if no indexing data structure is used, the complexity of the DBSCAN algorithm is O(n^2)

#### Strategy for parallelization

The DBSCAN algorithm is an inherently sequential algorithm. A quick look at the pseudo code provided in the original
paper will show why.

The parallelization strategy for DBSCAN involves 3 phases.
* Data distribution phase preserving spatial locality
* InCore DBSCAN Clustering
* Merging InCore clusters to form Global clusters

Few methods to parallelize DBSCAN have been proposed in the recent past and can be found [here](http://delivery.acm.org/10.1145/2390000/2389081/a62-patwary.pdf?ip=14.139.128.15&id=2389081&acc=ACTIVE%20SERVICE&key=045416EF4DDA69D9%2EDB7584019D0D7099%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=973347130&CFTOKEN=86904048&__acm__=1503663674_474f7b3eaa352d59c2905c1f530907c1) and [here](http://delivery.acm.org/10.1145/2840000/2834894/a2-gotz.pdf?ip=14.139.128.15&id=2834894&acc=ACTIVE%20SERVICE&key=045416EF4DDA69D9%2EDB7584019D0D7099%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=973347130&CFTOKEN=86904048&__acm__=1503663710_9001c4bdf3d4dc3ed6658d8a6aaf1190)

In order to minimize the communication costs for large datasets, we provide an approximate DBSCAN algorithm. Data division among nodes is done randomly
violating the spatial locality condition. Clustering is performed InCore and cluster merging is done by merging two clusters
that are nearby and that contain significant number of points.

### Parameters
<div class="table-striped">
  <table class="table">
    <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default Value</th>
    </tr>
    <tr>
        <td><code>'distanceMeasure</code></td>
        <td>The metric used for calculating distance, see <a href="../distance-metrics.html">Distance Metrics</a></td>
        <td><code>'Euclidean</code></td>
    </tr>
    <tr>
        <td><code>'Eps</code></td>
        <td>The radius that defines the neighbourhood of a point</code></td>
        <td>NA</td>
    </tr>
    <tr>
        <td><code>'Minpts</code></td>
        <td>The threshold for number of points in the neighbourhood of another point</code></td>
        <td>NA</td>
    </tr>
    </tr>
  </table>
</div>


### Example
    Upcoming!

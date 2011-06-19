/**
 * This package contains some common tasks for graph algorithms. It proposes a standard tool chain to process a graph:
 * <ol>
 * 	<li>Simplify the graph with {@link org.apache.mahout.graph.common.SimplifyGraphJob}. Parse a text file to a small representation of
 * 	edges without loops and duplicate edges. After this step the graph is interpreted as an undirected graph.</li>
 * 	<li>Augment the graph with vertex degrees using {@link AugmentGraphWithDegreesJob} which can be achieved in a  two-step MapReduce
 * 		pipeline.</li>
 * </ol>
 */
package org.apache.mahout.graph.common;

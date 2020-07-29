/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * <h2>In-memory mapreduce implementation of Random Decision Forests</h2>
 *
 * <p>Each mapper is responsible for growing a number of trees with a whole copy of the dataset loaded in memory,
 * it uses the reference implementation's code to build each tree and estimate the oob error.</p>
 *
 * <p>The dataset is distributed to the slave nodes using the {@link org.apache.hadoop.filecache.DistributedCache}.
 * A custom {@link org.apache.hadoop.mapreduce.InputFormat}
 * ({@link org.apache.mahout.classifier.df.mapreduce.inmem.InMemInputFormat}) is configured with the
 * desired number of trees and generates a number of {@link org.apache.hadoop.mapreduce.InputSplit}s
 * equal to the configured number of maps.</p>
 *
 * <p>There is no need for reducers, each map outputs (the trees it built and, for each tree, the labels the
 * tree predicted for each out-of-bag instance. This step has to be done in the mapper because only there we
 * know which instances are o-o-b.</p>
 *
 * <p>The Forest builder ({@link org.apache.mahout.classifier.df.mapreduce.inmem.InMemBuilder}) is responsible
 * for configuring and launching the job.
 * At the end of the job it parses the output files and builds the corresponding
 * {@link org.apache.mahout.classifier.df.DecisionForest}.</p>
 */
package org.apache.mahout.classifier.df.mapreduce.inmem;

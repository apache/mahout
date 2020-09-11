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
 * <p></p>This package provides several clustering algorithm implementations. Clustering usually groups a set of
 * objects into groups of similar items. The definition of similarity usually is up to you - for text documents,
 * cosine-distance/-similarity is recommended. Mahout also features other types of distance measure like
 * Euclidean distance.</p>
 *
 * <p></p>Input of each clustering algorithm is a set of vectors representing your items. For texts in general these are
 * <a href="http://en.wikipedia.org/wiki/TFIDF">TFIDF</a> or
 * <a href="http://en.wikipedia.org/wiki/Bag_of_words">Bag of words</a> representations of the documents.</p>
 *
 * <p>Output of each clustering algorithm is either a hard or soft assignment of items to clusters.</p>
 */
package org.apache.mahout.clustering;

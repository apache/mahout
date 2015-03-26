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

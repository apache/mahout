---
layout: doc-page
title: (Deprecated)  Viewing Results
---

<div style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 15px; border-radius: 5px; margin-bottom: 20px;"> <strong>⚠️ Deprecation Notice:</strong> This component (<em>Map Reduce</em>) is no longer actively maintained. It remains accessible for historical reference but is not recommended for new development.  Please check out <a href="https://mahout.apache.org/quantum-computing-primer/" style="color: #0c5460; text-decoration: underline;">Qumat - Mahout's Quantum Computing Primer</a> for the latest innovations.</div>

<a name="ViewingResults-Intro"></a>
# Intro

Many of the Mahout libraries run as batch jobs, dumping results into Hadoop
sequence files or other data structures.  This page is intended to
demonstrate the various ways one might inspect the outcome of various jobs.
 The page is organized by algorithms.

<a name="ViewingResults-GeneralUtilities"></a>
# General Utilities

<a name="ViewingResults-SequenceFileDumper"></a>
## Sequence File Dumper


<a name="ViewingResults-Clustering"></a>
# Clustering

<a name="ViewingResults-ClusterDumper"></a>
## Cluster Dumper

Run the following to print out all options:

    java  -cp "*" org.apache.mahout.utils.clustering.ClusterDumper --help



<a name="ViewingResults-Example"></a>
### Example

    java  -cp "*" org.apache.mahout.utils.clustering.ClusterDumper --seqFileDir
./solr-clust-n2/out/clusters-2
          --dictionary ./solr-clust-n2/dictionary.txt
          --substring 100 --pointsDir ./solr-clust-n2/out/points/
    



<a name="ViewingResults-ClusterLabels(MAHOUT-163)"></a>
## Cluster Labels (MAHOUT-163)

<a name="ViewingResults-Classification"></a>
# Classification

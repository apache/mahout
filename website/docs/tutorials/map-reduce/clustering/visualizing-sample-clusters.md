---
layout: tutorial
title: (Deprecated)  Visualizing Sample Clusters
theme:
   name: retro-mahout
---

<a name="VisualizingSampleClusters-Introduction"></a>
# Introduction

Mahout provides examples to visualize sample clusters that gets created by
our clustering algorithms. Note that the visualization is done by Swing programs. You have to be in a window system on the same
machine you run these, or logged in via a remote desktop.

For visualizing the clusters, you have to execute the Java
classes under *org.apache.mahout.clustering.display* package in
mahout-examples module. The easiest way to achieve this is to [setup Mahout](users/basics/quickstart.html) in your IDE.

<a name="VisualizingSampleClusters-Visualizingclusters"></a>
# Visualizing clusters

The following classes in *org.apache.mahout.clustering.display* can be run
without parameters to generate a sample data set and run the reference
clustering implementations over them:

1. **DisplayClustering** - generates 1000 samples from three, symmetric
distributions. This is the same data set that is used by the following
clustering programs. It displays the points on a screen and superimposes
the model parameters that were used to generate the points. You can edit
the *generateSamples()* method to change the sample points used by these
programs.
1. **DisplayClustering** - displays initial areas of generated points
1. **DisplayCanopy** - uses Canopy clustering
1. **DisplayKMeans** - uses k-Means clustering
1. **DisplayFuzzyKMeans** - uses Fuzzy k-Means clustering
1. **DisplaySpectralKMeans** - uses Spectral KMeans via map-reduce algorithm

If you are using Eclipse, just right-click on each of the classes mentioned above and choose "Run As -Java Application". To run these directly from the command line:

    cd $MAHOUT_HOME/examples
    mvn -q exec:java -Dexec.mainClass=org.apache.mahout.clustering.display.DisplayClustering

You can substitute other names above for *DisplayClustering*. 


Note that some of these programs display the sample points and then superimpose all of the clusters from each iteration. The last iteration's clusters are in
bold red and the previous several are colored (orange, yellow, green, blue,
magenta) in order after which all earlier clusters are in light grey. This
helps to visualize how the clusters converge upon a solution over multiple
iterations.
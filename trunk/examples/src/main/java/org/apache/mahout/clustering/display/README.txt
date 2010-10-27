The following classes can be run without parameters to generate a sample data set and 
run the reference clustering implementations over them:

DisplayClustering - generates 1000 samples from three, symmetric distributions. This is the same 
    data set that is used by the following clustering programs. It displays the points on a screen
    and superimposes the model parameters that were used to generate the points. You can edit the
    generateSamples() method to change the sample points used by these programs.
    
  * DisplayDirichlet - uses Dirichlet Process clustering
  * DisplayCanopy - uses Canopy clustering
  * DisplayKMeans - uses k-Means clustering
  * DisplayFuzzyKMeans - uses Fuzzy k-Means clustering
  * DisplayMeanShift - uses MeanShift clustering
  
  * NOTE: some of these programs display the sample points and then superimpose all of the clusters
    from each iteration. The last iteration's clusters are in bold red and the previous several are 
    colored (orange, yellow, green, blue, violet) in order after which all earlier clusters are in
    light grey. This helps to visualize how the clusters converge upon a solution over multiple
    iterations.
  * NOTE: by changing the parameter values (k, ALPHA_0, numIterations) and the display SIGNIFICANCE
    you can obtain different results.
    
  
    
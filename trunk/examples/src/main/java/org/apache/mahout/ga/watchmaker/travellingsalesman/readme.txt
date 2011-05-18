This example shows how it's easy to adapt an existing Watchmaker program to use Mahout.

The original code is from the Watchmaker Travelling Salesman example (https://watchmaker.dev.java.net/).

Nearly all the files are the same as in the original code, the only modifications are:

. StrategyPanel.java : EvolutionPanel class has been modified to add a "distributed (mahout)" JCheckBox.

. TravellingSalesman.java : (originally TravellingSalesmanApplet.java)  has been modified to add a main 
  method that runs the JApplet inside a modal JDialog, this way Hadoop waits until the Dialog is closed 
  to terminate.

. EvolutionaryTravellingSalesman.java : has been modified to use Mahout whenever requested by the GUI 
  (when the mahout checkbox is checked). To use Mahout we start by wrapping the original FitnessEvaluator
  inside a STFitnessEvaluator, and the pass it to a STEvolutionEngine, instead of a StandaloneEngine.

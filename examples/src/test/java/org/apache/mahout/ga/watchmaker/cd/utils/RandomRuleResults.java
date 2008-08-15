package org.apache.mahout.ga.watchmaker.cd.utils;

import java.util.HashMap;
import java.util.Map;

import org.apache.mahout.ga.watchmaker.cd.CDFitness;

public class RandomRuleResults {

  private static Map<Integer, CDFitness> results = new HashMap<Integer, CDFitness>();

  public static synchronized void addResult(int ruleid, CDFitness fit) {
    CDFitness f = results.get(ruleid);
    if (f == null)
      f = new CDFitness(fit);
    else
      f.add(fit);
    
    results.put(ruleid, f);
  }

  public static CDFitness getResult(int ruleid) {
    return results.get(ruleid);
  }
}

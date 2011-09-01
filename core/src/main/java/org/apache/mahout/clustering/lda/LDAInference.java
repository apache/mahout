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

package org.apache.mahout.clustering.lda;

import java.util.Iterator;

import org.apache.commons.math.special.Gamma;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleDoubleFunction;

/**
 * Class for performing infererence on a document, which involves computing (an approximation to)
 * p(word|topic) for each word and topic, and a prior distribution p(topic) for each topic.
 */
public class LDAInference {
  
  private static final double E_STEP_CONVERGENCE = 1.0E-6;
  private static final int MAX_ITER = 20;

  private DenseMatrix phi;
  private final LDAState state;

  public LDAInference(LDAState state) {
    this.state = state;
  }
  
  /**
   * An estimate of the probabilities for each document. Gamma(k) is the probability of seeing topic k in the
   * document, phi(k,w) is the (log) probability of topic k generating w in this document.
   */
  public static class InferredDocument {
    
    private final Vector wordCounts;
    private final Vector gamma; // p(topic)
    private final Matrix mphi; // log p(columnMap(w)|t)
    private final int[] columnMap; // maps words into the matrix's column map
    private final double logLikelihood;

    InferredDocument(Vector wordCounts, Vector gamma, int[] columnMap, Matrix phi, double ll) {
      this.wordCounts = wordCounts;
      this.gamma = gamma;
      this.mphi = phi;
      this.columnMap = columnMap;
      this.logLikelihood = ll;
    }

    public double phi(int k, int w) {
      return mphi.getQuick(k, columnMap[w]);
    }
    
    public Vector getWordCounts() {
      return wordCounts;
    }
    
    public Vector getGamma() {
      return gamma;
    }

    public double getLogLikelihood() {
      return logLikelihood;
    }
  }
  
  /**
   * Performs inference on the given document, returning an InferredDocument.
   */
  public InferredDocument infer(Vector wordCounts) {
    double docTotal = wordCounts.zSum();
    int docLength = wordCounts.size(); // cardinality of document vectors
    
    // initialize variational approximation to p(z|doc)
    Vector gamma = new DenseVector(state.getNumTopics());
    gamma.assign(state.getTopicSmoothing() + docTotal / state.getNumTopics());
    Vector nextGamma = new DenseVector(state.getNumTopics());
    createPhiMatrix(docLength);
    
    Vector digammaGamma = digammaGamma(gamma);
    
    int[] map = new int[docLength];
    
    int iteration = 0;
    
    boolean converged = false;
    double oldLL = 1.0;
    while (!converged && iteration < MAX_ITER) {
      nextGamma.assign(state.getTopicSmoothing()); // nG := alpha, for all topics
      
      int mapping = 0;
      for (Iterator<Vector.Element> iter = wordCounts.iterateNonZero(); iter.hasNext();) {
        Vector.Element e = iter.next();
        int word = e.index();
        Vector phiW = eStepForWord(word, digammaGamma);
        phi.assignColumn(mapping, phiW);
        if (iteration == 0) { // first iteration
          map[word] = mapping;
        }
        
        for (int k = 0; k < nextGamma.size(); ++k) {
          double g = nextGamma.getQuick(k);
          nextGamma.setQuick(k, g + e.get() * Math.exp(phiW.getQuick(k)));
        }
        
        mapping++;
      }
      
      Vector tempG = gamma;
      gamma = nextGamma;
      nextGamma = tempG;
      
      digammaGamma = digammaGamma(gamma);
      
      double ll = computeLikelihood(wordCounts, map, phi, gamma, digammaGamma);
      // isNotNaNAssertion(ll);
      converged = oldLL < 0.0 && (oldLL - ll) / oldLL < E_STEP_CONVERGENCE;
      
      oldLL = ll;
      iteration++;
    }
    
    return new InferredDocument(wordCounts, gamma, map, phi, oldLL);
  }

  /**
   * @param gamma
   * @return a vector whose entries are digamma(oldEntry) - digamma(gamma.zSum())
   */
  private Vector digammaGamma(Vector gamma) {
    // digamma is expensive, precompute
    Vector digammaGamma = digamma(gamma);
    // and log normalize:
    double digammaSumGamma = digamma(gamma.zSum());
    for (int i = 0; i < state.getNumTopics(); i++) {
      digammaGamma.setQuick(i, digammaGamma.getQuick(i) - digammaSumGamma);
    }
    return digammaGamma;
  }
  
  private void createPhiMatrix(int docLength) {
    if (phi == null || phi.rowSize() != docLength) {
      phi = new DenseMatrix(state.getNumTopics(), docLength);
    } else {
      phi.assign(0);
    }
  }

  /**
   * diGamma(x) = gamma'(x)/gamma(x)
   * logGamma(x) = log(gamma(x))
   *
   * ll = log(gamma(smooth*numTop) / smooth^numTop) +
   *   sum_{i < numTop} (smooth - g[i])*(digamma(g[i]) - digamma(|g|)) + log(gamma(g[i])
   * Computes the log likelihood of the wordCounts vector, given \phi, \gamma, and \digamma(gamma)
   * @param wordCounts
   * @param map
   * @param phi
   * @param gamma
   * @param digammaGamma
   * @return
   */
  private double computeLikelihood(Vector wordCounts, int[] map, Matrix phi, Vector gamma, Vector digammaGamma) {
    double ll = 0.0;
    
    // log normalizer for q(gamma);
    ll += Gamma.logGamma(state.getTopicSmoothing() * state.getNumTopics());
    ll -= state.getNumTopics() * Gamma.logGamma(state.getTopicSmoothing());
    // isNotNaNAssertion(ll);
    
    // now for the the rest of q(gamma);
    for (int k = 0; k < state.getNumTopics(); ++k) {
      double gammaK = gamma.get(k);
      ll += (state.getTopicSmoothing() - gammaK) * digammaGamma.getQuick(k);
      ll += Gamma.logGamma(gammaK);
      
    }
    ll -= Gamma.logGamma(gamma.zSum());
    // isNotNaNAssertion(ll);
    
    // for each word
    for (Iterator<Vector.Element> iter = wordCounts.iterateNonZero(); iter.hasNext();) {
      Vector.Element e = iter.next();
      int w = e.index();
      double n = e.get();
      int mapping = map[w];
      // now for each topic:
      for (int k = 0; k < state.getNumTopics(); k++) {
        double llPart = 0.0;
        double phiKMapping = phi.getQuick(k, mapping);
        llPart += Math.exp(phiKMapping)
                  * (digammaGamma.getQuick(k) - phiKMapping + state.logProbWordGivenTopic(w, k));
        
        ll += llPart * n;
        
        // likelihoodAssertion(w, k, llPart);
      }
    }
    // isLessThanOrEqualsZero(ll);
    return ll;
  }
  
  /**
   * Compute log q(k|w,doc) for each topic k, for a given word.
   */
  private Vector eStepForWord(int word, Vector digammaGamma) {
    Vector phi = new DenseVector(state.getNumTopics()); // log q(k|w), for each w
    double phiTotal = Double.NEGATIVE_INFINITY; // log Normalizer
    for (int k = 0; k < state.getNumTopics(); ++k) { // update q(k|w)'s param phi
      phi.setQuick(k, state.logProbWordGivenTopic(word, k) + digammaGamma.getQuick(k));
      phiTotal = LDAUtil.logSum(phiTotal, phi.getQuick(k));
      
      // assertions(word, digammaGamma, phiTotal, k);
    }
    for (int i = 0; i < state.getNumTopics(); i++) {
      phi.setQuick(i, phi.getQuick(i) - phiTotal); // log normalize
    }
    return phi;
  }
  
  private static Vector digamma(Vector v) {
    Vector digammaGamma = new DenseVector(v.size());
    digammaGamma.assign(v, new DoubleDoubleFunction() {
      @Override
      public double apply(double unused, double g) {
        return digamma(g);
      }
    });
    return digammaGamma;
  }

  /**
   * Approximation to the digamma function, from Radford Neal.
   * 
   * Original License: Copyright (c) 1995-2003 by Radford M. Neal
   * 
   * Permission is granted for anyone to copy, use, modify, or distribute this program and accompanying
   * programs and documents for any purpose, provided this copyright notice is retained and prominently
   * displayed, along with a note saying that the original programs are available from Radford Neal's web
   * page, and note is made of any changes made to the programs. The programs and documents are distributed
   * without any warranty, express or implied. As the programs were written for research purposes only, they
   * have not been tested to the degree that would be advisable in any important application. All use of these
   * programs is entirely at the user's own risk.
   * 
   * 
   * Ported to Java for Mahout.
   * 
   */
  private static double digamma(double x) {
    double r = 0.0;
    
    while (x <= 5) {
      r -= 1 / x;
      x += 1;
    }
    
    double f = 1.0 / (x * x);
    double t = f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0
        + f * (-1.0 / 132.0 + f * (691.0 / 32760.0 + f * (-1.0 / 12.0 + f * 3617.0 / 8160.0)))))));
    return r + Math.log(x) - 0.5 / x + t;
  }

}

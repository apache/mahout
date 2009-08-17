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

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.math.special.Gamma;
import org.apache.mahout.matrix.BinaryFunction;
import org.apache.mahout.matrix.DenseMatrix;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Matrix;
import org.apache.mahout.matrix.Vector;


/**
* Class for performing infererence on a document, which involves
* computing (an approximation to) p(word|topic) for each word and
* topic, and a prior distribution p(topic) for each topic.
*/
public class LDAInference {
  public LDAInference(LDAState state) {
    this.state = state;
  }

  /**
  * An estimate of the probabilitys for each document.
  * Gamma(k) is the probability of seeing topic k in
  * the document, phi(k,w) is the probability of
  * topic k generating w in this document.
  */
  public class InferredDocument {

    public final Vector wordCounts;
    public final Vector gamma; // p(topic)
    private final Matrix mphi; // log p(columnMap(w)|t)
    private final Map<Integer, Integer> columnMap; // maps words into the matrix's column map
    public final double logLikelihood;

    public double phi(int k, int w) {
      return mphi.getQuick(k, columnMap.get(w));
    }

    InferredDocument(Vector wordCounts, Vector gamma,
                     Map<Integer, Integer> columnMap, Matrix phi,
                     double ll) {
      this.wordCounts = wordCounts;
      this.gamma = gamma;
      this.mphi = phi;
      this.columnMap = columnMap;
      this.logLikelihood = ll;
    }
  }

  /**
  * Performs inference on the given document, returning
  * an InferredDocument.
  */
  public InferredDocument infer(Vector wordCounts) {
    double docTotal = wordCounts.zSum();
    int docLength = wordCounts.size();

    // initialize variational approximation to p(z|doc)
    Vector gamma = new DenseVector(state.numTopics);
    gamma.assign(state.topicSmoothing + docTotal / state.numTopics);
    Vector nextGamma = new DenseVector(state.numTopics);

    DenseMatrix phi = new DenseMatrix(state.numTopics, docLength);

    boolean converged = false;
    double oldLL = 1;
    // digamma is expensive, precompute
    Vector digammaGamma = digamma(gamma);
    // and log normalize:
    double digammaSumGamma = digamma(gamma.zSum());
    digammaGamma = digammaGamma.plus(-digammaSumGamma);

    Map<Integer, Integer> columnMap = new HashMap<Integer, Integer>();

    int iteration = 0;
    final int MAX_ITER = 20;

    while (!converged && iteration < MAX_ITER) {
      nextGamma.assign(state.topicSmoothing); // nG := alpha, for all topics

      int mapping = 0;
      for (Iterator<Vector.Element> iter = wordCounts.iterateNonZero();
          iter.hasNext();) {
      Vector.Element e = iter.next();
        int word = e.index();
        Vector phiW = eStepForWord(word, digammaGamma);
        phi.assignColumn(mapping, phiW);
        if (iteration == 0) { // first iteration
          columnMap.put(word, mapping);
        }

        for (int k = 0; k < nextGamma.size(); ++k) {
          double g = nextGamma.getQuick(k);
          nextGamma.setQuick(k, g + e.get() * Math.exp(phiW.get(k)));
        }

        mapping++;
      }

      Vector tempG = gamma;
      gamma = nextGamma;
      nextGamma = tempG;

      // digamma is expensive, precompute
      digammaGamma = digamma(gamma);
      // and log normalize:
      digammaSumGamma = digamma(gamma.zSum());
      digammaGamma = digammaGamma.plus(-digammaSumGamma);

      double ll = computeLikelihood(wordCounts, columnMap, phi, gamma, digammaGamma);
      converged = oldLL < 0 && ((oldLL - ll) / oldLL < E_STEP_CONVERGENCE);
      assert !Double.isNaN(ll);

      oldLL = ll;
      iteration++;
    }

    return new InferredDocument(wordCounts, gamma, columnMap, phi, oldLL);
  }

  private LDAState state;

  private double computeLikelihood(Vector wordCounts, Map<Integer, Integer> columnMap,
      Matrix phi, Vector gamma, Vector digammaGamma) {
    double ll = 0.0;

    // log normalizer for q(gamma);
    ll += Gamma.logGamma(state.topicSmoothing * state.numTopics);
    ll -= state.numTopics * Gamma.logGamma(state.topicSmoothing);
    assert !Double.isNaN(ll) : state.topicSmoothing + " " + state.numTopics;

    // now for the the rest of q(gamma);
    for (int k = 0; k < state.numTopics; ++k) {
      ll += (state.topicSmoothing - gamma.get(k)) * digammaGamma.get(k);
      ll += Gamma.logGamma(gamma.get(k));

    }
    ll -= Gamma.logGamma(gamma.zSum());
    assert !Double.isNaN(ll);


    // for each word
    for (Iterator<Vector.Element> iter = wordCounts.iterateNonZero();
        iter.hasNext();) {
      Vector.Element e = iter.next();
      int w = e.index();
      double n = e.get();
      int mapping = columnMap.get(w);
      // now for each topic:
      for (int k = 0; k < state.numTopics; k++) {
        double llPart = 0.0;
        llPart += Math.exp(phi.get(k, mapping))
          * (digammaGamma.get(k) - phi.get(k, mapping)
             + state.logProbWordGivenTopic(w, k));

        ll += llPart * n;

        assert state.logProbWordGivenTopic(w, k)  < 0;
        assert !Double.isNaN(llPart);
      }
    }
    assert ll <= 0;
    return ll;
  }

  /**
   * Compute log q(k|w,doc) for each topic k, for a given word.
   */
  private Vector eStepForWord(int word, Vector digammaGamma) {
    Vector phi = new DenseVector(state.numTopics); // log q(k|w), for each w
    double phiTotal = Double.NEGATIVE_INFINITY; // log Normalizer
    for (int k = 0; k < state.numTopics; ++k) { // update q(k|w)'s param phi
      phi.set(k, state.logProbWordGivenTopic(word, k) + digammaGamma.get(k));
      phiTotal = LDAUtil.logSum(phiTotal, phi.get(k));

      assert !Double.isNaN(phiTotal);
      assert !Double.isNaN(state.logProbWordGivenTopic(word, k));
      assert !Double.isInfinite(state.logProbWordGivenTopic(word, k));
      assert !Double.isNaN(digammaGamma.get(k));
    }
    return phi.plus(-phiTotal); // log normalize
  }


  private static Vector digamma(Vector v) {
    Vector digammaGamma = new DenseVector(v.size());
    digammaGamma.assign(v, new BinaryFunction() {
      public double apply(double unused, double g) {
        return digamma(g);
      }
    });
    return digammaGamma;
  }

  /**
   * Approximation to the digamma function, from Radford Neal.
   *
   * Original License:
   * Copyright (c) 1995-2003 by Radford M. Neal
   *
   * Permission is granted for anyone to copy, use, modify, or distribute this
   * program and accompanying programs and documents for any purpose, provided
   * this copyright notice is retained and prominently displayed, along with
   * a note saying that the original programs are available from Radford Neal's
   * web page, and note is made of any changes made to the programs.  The
   * programs and documents are distributed without any warranty, express or
   * implied.  As the programs were written for research purposes only, they have
   * not been tested to the degree that would be advisable in any important
   * application.  All use of these programs is entirely at the user's own risk.
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

    double f = 1. / (x * x);
    double t = f * (-1 / 12.0
        + f * (1 / 120.0
        + f * (-1 / 252.0
        + f * (1 / 240.0 
        + f * (-1 / 132.0 
        + f * (691 / 32760.0 
        + f * (-1 / 12.0 
        + f * 3617.0 / 8160.0)))))));
    return r + Math.log(x) - 0.5 / x + t;
  }

    private static final double E_STEP_CONVERGENCE = 1E-6;
}

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

package org.apache.mahout.cf.taste.hadoop.similarity;

import java.util.Iterator;

import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;

/**
 * Distributed version of {@link TanimotoCoefficientSimilarity}
 */
public class DistributedTanimotoCoefficientSimilarity extends AbstractDistributedItemSimilarity {

	@Override
	protected double doComputeResult(Iterator<CoRating> coratings,
			double weightOfItemVectorX, double weightOfItemVectorY,
			int numberOfUsers) {

	  double preferringXAndY = 0;
	  while (coratings.hasNext()) {
	    coratings.next();
	    preferringXAndY++;
	  }

	  if (preferringXAndY == 0) {
	    return Double.NaN;
	  }

	  return (preferringXAndY / (weightOfItemVectorX + weightOfItemVectorY - preferringXAndY));
	}

	@Override
	public double weightOfItemVector(Iterator<Float> prefValues) {
		double nonZeroEntries = 0;
		while (prefValues.hasNext()) {
		  prefValues.next();
		  nonZeroEntries++;
		}
		return nonZeroEntries;
	}

}

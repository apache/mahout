/**
 * <p>Provides basic evolutionary optimization using <a href="http://arxiv.org/abs/0803.3838">recorded-step</a>
 * mutation.</p>
 *
 * <p>With this style of optimization, we can optimize a function {@code f: R^n -> R} by stochastic
 * hill-climbing with some of the benefits of conjugate gradient style history encoded in the mutation function.
 * This mutation function will adapt to allow weakly directed search rather than using the somewhat more
 * conventional symmetric Gaussian.</p>
 *
 * <p>With recorded-step mutation, the meta-mutation parameters are all auto-encoded in the current state of each point.
 * This avoids the classic problem of having more mutation rate parameters than are in the original state and then
 * requiring even more parameters to describe the meta-mutation rate. Instead, we store the previous point and one
 * omni-directional mutation component. Mutation is performed by first mutating along the line formed by the previous
 * and current points and then adding a scaled symmetric Gaussian.  The magnitude of the omni-directional mutation is
 * then mutated using itself as a scale.</p>
 *
 * <p>Because it is convenient to not restrict the parameter space, this package also provides convenient parameter
 * mapping methods.  These mapping methods map the set of reals to a finite open interval (a,b) in such a way that
 * {@code lim_{x->-\inf} f(x) = a} and {@code lim_{x->\inf} f(x) = b}. The linear mapping is defined so that
 * {@code f(0) = (a+b)/2} and the exponential mapping requires that a and b are both positive and has
 * {@code f(0) = sqrt(ab)}. The linear mapping is useful for values that must stay roughly within a range but
 * which are roughly uniform within the center of that range. The exponential
 * mapping is useful for values that must stay within a range but whose distribution is roughly exponential near
 * geometric mean of the end-points.  An identity mapping is also supplied.</p>
 */
package org.apache.mahout.ep;

---
layout: algorithm
title: Durbin-Watson Test
theme:
    name: mahout2
---

### About

The [Durbin Watson Test](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic) is a test for serial correlation
in error terms.  The Durbin Watson test statistic <foo>\(d\)</foo> can take values between 0 and 4, and in general

- <foo>\(d \lt 1.5 \)</foo> implies positive autocorrelation
- <foo>\(d \gt 2.5 \)</foo>  implies negative autocorrelation
- <foo>\(1.5 \lt d \lt 2.5 \)</foo>  implies to autocorrelation.

Implementation is based off of the `durbinWatsonTest` function in the [`car`](https://cran.r-project.org/web/packages/car/index.html) package in R

### Parameters

### Example

#### R Prototype
    
    library(car)
    residuals <- seq(0, 4.9, 0.1)
    ## perform Durbin-Watson test
    durbinWatsonTest(residuals)
    
#### In Apache Mahout

    
    // A DurbinWatson Test must be performed on a model. The model does not matter.
    val drmX = drmParallelize( dense((0 until 50).toArray.map( t => Math.pow(-1.0, t)) ) ).t
    val drmY = drmX + err1 + 1
    var model = new OrdinaryLeastSquares[Int]().fit(drmX, drmY)
    // end arbitrary model 
    
    val err1 =  drmParallelize( dense((0.0 until 5.0 by 0.1).toArray) ).t
    val syntheticResiduals = err1
    model = AutocorrelationTests.DurbinWatson(model, syntheticResiduals)
    val myAnswer: Double = model.testResults.getOrElse('durbinWatsonTestStatistic, -1.0).asInstanceOf[Double]


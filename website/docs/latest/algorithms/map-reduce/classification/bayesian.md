<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
---
layout: doc-page
title: (Deprecated) 

    
---

# Naive Bayes


## Intro

Mahout currently has two Naive Bayes Map-Reduce implementations.  The first is standard Multinomial Naive Bayes. The second is an implementation of Transformed Weight-normalized Complement Naive Bayes as introduced by Rennie et al. [[1]](http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf). We refer to the former as Bayes and the latter as CBayes.

Where Bayes has long been a standard in text classification, CBayes is an extension of Bayes that performs particularly well on datasets with skewed classes and has been shown to be competitive with algorithms of higher complexity such as Support Vector Machines. 


## Implementations
Both Bayes and CBayes are currently trained via MapReduce Jobs. Testing and classification can be done via a MapReduce Job or sequentially.  Mahout provides CLI drivers for preprocessing, training and testing. A Spark implementation is currently in the works ([MAHOUT-1493](https://issues.apache.org/jira/browse/MAHOUT-1493)).

## Preprocessing and Algorithm

As described in [[1]](http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) Mahout Naive Bayes is broken down into the following steps (assignments are over all possible index values):  

- Let `\(\vec{d}=(\vec{d_1},...,\vec{d_n})\)` be a set of documents; `\(d_{ij}\)` is the count of word `\(i\)` in document `\(j\)`.
- Let `\(\vec{y}=(y_1,...,y_n)\)` be their labels.
- Let `\(\alpha_i\)` be a smoothing parameter for all words in the vocabulary; let `\(\alpha=\sum_i{\alpha_i}\)`. 
- **Preprocessing**(via seq2Sparse) TF-IDF transformation and L2 length normalization of `\(\vec{d}\)`
    1. `\(d_{ij} = \sqrt{d_{ij}}\)` 
    2. `\(d_{ij} = d_{ij}\left(\log{\frac{\sum_k1}{\sum_k\delta_{ik}+1}}+1\right)\)` 
    3. `\(d_{ij} =\frac{d_{ij}}{\sqrt{\sum_k{d_{kj}^2}}}\)` 
- **Training: Bayes**`\((\vec{d},\vec{y})\)` calculate term weights `\(w_{ci}\)` as:
    1. `\(\hat\theta_{ci}=\frac{d_{ic}+\alpha_i}{\sum_k{d_{kc}}+\alpha}\)`
    2. `\(w_{ci}=\log{\hat\theta_{ci}}\)`
- **Training: CBayes**`\((\vec{d},\vec{y})\)` calculate term weights `\(w_{ci}\)` as:
    1. `\(\hat\theta_{ci} = \frac{\sum_{j:y_j\neq c}d_{ij}+\alpha_i}{\sum_{j:y_j\neq c}{\sum_k{d_{kj}}}+\alpha}\)`
    2. `\(w_{ci}=-\log{\hat\theta_{ci}}\)`
    3. `\(w_{ci}=\frac{w_{ci}}{\sum_i \lvert w_{ci}\rvert}\)`
- **Label Assignment/Testing:**
    1. Let `\(\vec{t}= (t_1,...,t_n)\)` be a test document; let `\(t_i\)` be the count of the word `\(t\)`.
    2. Label the document according to `\(l(t)=\arg\max_c \sum\limits_{i} t_i w_{ci}\)`

As we can see, the main difference between Bayes and CBayes is the weight calculation step.  Where Bayes weighs terms more heavily based on the likelihood that they belong to class `\(c\)`, CBayes seeks to maximize term weights on the likelihood that they do not belong to any other class.  

## Running from the command line

Mahout provides CLI drivers for all above steps.  Here we will give a simple overview of Mahout CLI commands used to preprocess the data, train the model and assign labels to the training set. An [example script](https://github.com/apache/mahout/blob/master/examples/bin/classify-20newsgroups.sh) is given for the full process from data acquisition through classification of the classic [20 Newsgroups corpus](https://mahout.apache.org/users/classification/twenty-newsgroups.html).  

- **Preprocessing:**
For a set of Sequence File Formatted documents in PATH_TO_SEQUENCE_FILES the [mahout seq2sparse](https://mahout.apache.org/users/basics/creating-vectors-from-text.html) command performs the TF-IDF transformations (-wt tfidf option) and L2 length normalization (-n 2 option) as follows:

        mahout seq2sparse 
          -i ${PATH_TO_SEQUENCE_FILES} 
          -o ${PATH_TO_TFIDF_VECTORS} 
          -nv 
          -n 2
          -wt tfidf

- **Training:**
The model is then trained using `mahout trainnb` .  The default is to train a Bayes model. The -c option is given to train a CBayes model:

        mahout trainnb
          -i ${PATH_TO_TFIDF_VECTORS} 
          -o ${PATH_TO_MODEL}/model 
          -li ${PATH_TO_MODEL}/labelindex 
          -ow 
          -c

- **Label Assignment/Testing:**
Classification and testing on a holdout set can then be performed via `mahout testnb`. Again, the -c option indicates that the model is CBayes.  The -seq option tells `mahout testnb` to run sequentially:

        mahout testnb 
          -i ${PATH_TO_TFIDF_TEST_VECTORS}
          -m ${PATH_TO_MODEL}/model 
          -l ${PATH_TO_MODEL}/labelindex 
          -ow 
          -o ${PATH_TO_OUTPUT} 
          -c 
          -seq

## Command line options

- **Preprocessing:**
  
  Only relevant parameters used for Bayes/CBayes as detailed above are shown. Several other transformations can be performed by `mahout seq2sparse` and used as input to Bayes/CBayes.  For a full list of `mahout seq2Sparse` options see the [Creating vectors from text](https://mahout.apache.org/users/basics/creating-vectors-from-text.html) page.

        mahout seq2sparse                         
          --output (-o) output             The directory pathname for output.        
          --input (-i) input               Path to job input directory.              
          --weight (-wt) weight            The kind of weight to use. Currently TF   
                                               or TFIDF. Default: TFIDF                  
          --norm (-n) norm                 The norm to use, expressed as either a    
                                               float or "INF" if you want to use the     
                                               Infinite norm.  Must be greater or equal  
                                               to 0.  The default is not to normalize    
          --overwrite (-ow)                If set, overwrite the output directory    
          --sequentialAccessVector (-seq)  (Optional) Whether output vectors should  
                                               be SequentialAccessVectors. If set true   
                                               else false                                
          --namedVector (-nv)              (Optional) Whether output vectors should  
                                               be NamedVectors. If set true else false   

- **Training:**

        mahout trainnb
          --input (-i) input               Path to job input directory.                 
          --output (-o) output             The directory pathname for output.                    
          --alphaI (-a) alphaI             Smoothing parameter. Default is 1.0
          --trainComplementary (-c)        Train complementary? Default is false.                        
          --labelIndex (-li) labelIndex    The path to store the label index in         
          --overwrite (-ow)                If present, overwrite the output directory   
                                               before running job                           
          --help (-h)                      Print out help                               
          --tempDir tempDir                Intermediate output directory                
          --startPhase startPhase          First phase to run                           
          --endPhase endPhase              Last phase to run

- **Testing:**

        mahout testnb   
          --input (-i) input               Path to job input directory.                  
          --output (-o) output             The directory pathname for output.            
          --overwrite (-ow)                If present, overwrite the output directory    
                                               before running job                                                

      
          --model (-m) model               The path to the model built during training   
          --testComplementary (-c)         Test complementary? Default is false.                          
          --runSequential (-seq)           Run sequential?                               
          --labelIndex (-l) labelIndex     The path to the location of the label index   
          --help (-h)                      Print out help                                
          --tempDir tempDir                Intermediate output directory                 
          --startPhase startPhase          First phase to run                            
          --endPhase endPhase              Last phase to run  


## Examples

Mahout provides an example for Naive Bayes classification:

1. [Classify 20 Newsgroups](twenty-newsgroups.html)
 
## References

[1]: Jason D. M. Rennie, Lawerence Shih, Jamie Teevan, David Karger (2003). [Tackling the Poor Assumptions of Naive Bayes Text Classifiers](http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf). Proceedings of the Twentieth International Conference on Machine Learning (ICML-2003).



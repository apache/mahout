---
layout: page
theme: 
    name: mahout2
---


<div class="container-fluid">
    <div class="row">
        <div class="col-md-8">
            <div class="row">
                <div class="col-s-12">
                <div class="mahoutBox1">
                    <h4> Apache Mahout(TM) is a <b>distributed linear algebra framework</b> and <b>mathematically expressive Scala DSL</b>
                    designed to let mathematicians, statisticians, and data scientists quickly <i>implementent their own algorithms</i>. 
                    Apache Spark is the reccomended out-of-the-box distributed back-end, <i>or can be extended to other distributed backends.</i></h4> 
                </div></div>
            </div> <!-- row --> 
            <div class="row">
                <div class="col-xs-4">
                <div class="mahoutBox3"><b>Mathematically Expressive Scala DSL</b>
                </div></div>
                <div class="col-xs-4">
                <div class="mahoutBox2"><b>Support for Multiple Distributed Backends (including Apache Spark)</b>
                </div></div>
                <div class="col-xs-4">
                <div class="mahoutBox2"><b>Modular Native Solvers for CPU/GPU/CUDA Acceleration</b>
                </div></div>
            </div> <!-- row --> 
        </div>
        <div class="col-md-4"> 
            <div class="mahoutBox2 col-md-11">
            <div class='jekyll-twitter-plugin'>
                <a class="twitter-timeline" data-width="300" data-height="300" data-tweet-limit="4" data-chrome="nofooter" href="https://twitter.com/ApacheMahout">Tweets by ApacheMahout</a>
                <script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
            </div></div>
        </div>
    </div>   
</div>

<!--
<div class="container">
    <div class="row">
        <div class="col-9">
            <div class="mahoutBox1">
            <h4> Apache Mahout(TM) is a <b>distributed linear algebra framework</b> and <b>mathematically expressive Scala DSL</b>
             designed to let mathematicians, statisticians, and data scientists quickly <i>implementent their own algorithms</i>. 
             Apache Spark is the reccomended out-of-the-box distributed back-end, <i>or can be extended to other distributed backends.</i></h4>
            </div>
        </div> <!-- col9 -->
<!--
<div class="col-3">
    <div class="row">
        <div class="col-md-12 col-sm-12 col-xs-12 text-center">
            <div class='jekyll-twitter-plugin'><a class="twitter-timeline" data-width="500" data-tweet-limit="4" data-chrome="nofooter" href="https://twitter.com/ApacheMahout">Tweets by ApacheMahout</a>
                <script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script></div>
            </div>
        <div class="col-md-12 col-sm-12 col-xs-12 text-center twitterBtn">
            <p style="text-align:center; margin-top: 32px; font-size: 12px; color: gray; font-weight: 200; font-style: italic; padding-bottom: 0;">See more tweets or</p>
            <a href="https://twitter.com/ApacheMahout" target="_blank" class="btn btn-primary btn-lg round" role="button">Follow Mahout on &nbsp;<i class="fa fa-twitter fa-lg" aria-hidden="true"></i></a>
        </div>
    </div>
</div>
</div> 
<div class="row">

<!-- Jumbotron -->
    
<!--
  <div class="newMahout col-md-4 col-sm-4">
    <h4>Simple and <br/>Extensible</h4>
    <div class="viz">
      <p>
        Build your own algorithms using Mahouts R like interface.  See an example in this 
        <a href="" target="_blank">demo</a>
      </p>
    </div>
  </div>
  <div class="newMahout col-md-4 col-sm-4">
    <h4>Support for Multiple <br/>Distributed Backends</h4>
    <div class="multi">
    <p>
       Custom bindings for Spark, Flink, and H20 enable a write once run anywhere machine learning platform
    </p>
    </div>
  </div>
  <div class="newMahout col-md-4 col-sm-4">
    <h4>Introducing Samsara an R<br/> dsl for writing ML algos</h4>
    <div class="personal">
    <p>
      Use this capability to write algorithms at scale, that will run on any backend 
    </p>
    </div>
  </div>
</div>
<div class="border row">
  <div class="newMahout col-md-4 col-sm-4">
    <h4>Support for GPUs</h4>
    <p>
      Distributed GPU Matrix-Matrix and Matrix-Vector multiplication on Spark along with sparse and dense matrix GPU-backed support.
    </p>
  </div>
  <div class="newMahout col-md-4 col-sm-4">
    <h4>Extensible Algorithms Framework</h4>
    <p>
       A new scikit-learn-like framework for algorithms with the goal for
       creating a consistent API for various machine-learning algorithms
    </p>
  </div>
  <div class="newMahout col-md-4 col-sm-4">
    <h4>0.13.1 - Future Plans</h4>
    <p>
      - JCuda native solver <br>
      - Scala 2.11 / Spark 2.x Support  <br>
      - Expaned Algorithms Framework
    </p>
  </div>
</div>
<div class="col-md-12 col-sm-12 col-xs-12 text-center">
  <p style="text-align:center; margin-top: 32px; font-size: 14px; color: gray; font-weight: 200; font-style: italic; padding-bottom: 0;">See more details in 
    <a href="tbd">0.13.0 Release Note</a>
  </p>
</div>

<aside>
    <div class="col-md-12 col-sm-6">
        <h2>Mahout Blogs</h2>
        {% for post in paginator.posts %}
        {% include tile.html %}
        {% endfor %}
    </div>  
    <div class="container col-sm-6 col-md-12">
        <h2>Mahout on Twitter</h2>
        <br/>
        <div class="row">
            <div class="col-md-12 col-sm-12 col-xs-12 text-center">
                <div class='jekyll-twitter-plugin'><a class="twitter-timeline" data-width="500" data-tweet-limit="4" data-chrome="nofooter" href="https://twitter.com/ApacheMahout">Tweets by ApacheMahout</a>
                    <script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script></div>
                </div>
            <div class="col-md-12 col-sm-12 col-xs-12 text-center twitterBtn">
                <p style="text-align:center; margin-top: 32px; font-size: 12px; color: gray; font-weight: 200; font-style: italic; padding-bottom: 0;">See more tweets or</p>
                <a href="https://twitter.com/ApacheMahout" target="_blank" class="btn btn-primary btn-lg round" role="button">Follow Mahout on &nbsp;<i class="fa fa-twitter fa-lg" aria-hidden="true"></i></a>
            </div>
        </div>
     </div>
</aside>
f-->


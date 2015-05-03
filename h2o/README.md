# Introduction

This document demonstrates the integration between Mahout (http://mahout.apache.org) and H2O (http://www.h2o.ai). The integration provides a H2O backend to the Mahout algebra DSL (similar to the Spark backend.)

## Setup

The integration depends on h2o-core maven artifact. This can either be fetched automatically through sonatype, or can be installed locally from source (run 'gradle install -x test' in http://github.com/0xdata/h2o-dev)

## Test

The integration with H2O can be used in either a local mode (single node) or a clustered mode.

### Simple (single node/local) test

Testing in local mode is pretty straight forward. Just run 'mvn test' as shown below.

    sh:~/mahout$ cd h2o
    sh:~/mahout/h2o$ mvn test
    ...
    ...
    All tests passed.
    ...
    sh:~/mahout/h2o$

### Distributed test

H2O is fundamentally a peer-to-peer system. H2O nodes join together to form a cloud on which high performance distributed math can be executed. Each node joins a cloud of a given name. Multiple clouds can exist on the same network at the same time as long as their names are different. Multiple nodes can exist on the same server as well (even belonging to the same cloud.)

The Mahout H2O integration is fit into this model by having N-1 "worker" nodes and one driver node, all belonging to the same cloud name. The default cloud name used for the integration is "mah2out". Clouds have to be spun up per task/job.

**WARNING**: Some Linux systems have default firewall rules which might block traffic required for the following tests. In order to successfully run the tests you might need to temporarily turn off firewall rules with `sh# iptables -F`

First bring up worker nodes:

    host-1:~/mahout$ ./bin/mahout h2o-node
    ...
    .. INFO: Cloud of size 1 formed [/W.X.Y.Z:54321]

Similarly,

    host-2:~/mahout$ ./bin/mahout h2o-node
    ...
    .. INFO: Cloud of size 2 formed [/A.B.C.D:54322]

... and so on. For the purpose of testing multiple (even all) instances can be run on the same system too.

The nodes discover each other over a multicast channel and establish consensus with Paxos. Next, start the driver just like running in local mode.

    host-N:~/mahout/h2o$ mvn test
    ...
    .. INFO: Cloud of size 3 formed [/E.F.G.H:54323]
    ...
    All tests passed.
    ...
    host-N:~/mahout/h2o$

The workers have to be restarted when when the driver node terminates (automating this is a future task.)
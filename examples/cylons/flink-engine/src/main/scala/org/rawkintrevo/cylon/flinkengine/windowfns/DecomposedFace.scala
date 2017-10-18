package org.rawkintrevo.cylon.flinkengine.windowfns

import org.apache.mahout.math.Vector

case class DecomposedFace(key: String
                          ,h : Int
                          ,w : Int
                          ,x : Int
                          ,y : Int
                          ,frame : Int
                          ,v : Vector
                          ,metaVec : Vector
                         , cluster: Int = -1
                         , name: String = "ghost"
                         , distanceFromCenter: Double = 0.0){

}

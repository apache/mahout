package org.apache.mahout.math.scalaframes

import org.scalatest.FunSuite
import org.apache.mahout.test.MahoutSuite
import java.nio.ByteBuffer
import scala.util.Random
import concurrent._
import scala.concurrent.duration.Duration
import sun.misc.Unsafe
import java.lang.reflect.Field

class ScalaframesSuite extends FunSuite with MahoutSuite {

  test("mutate") {
    val testFrame = new BaseDFrame()

    val mutatedFrame = testFrame.mutate(
      "ACol" := col("5") + col(4),
      "BCol" := col("AAA") + 3
    )
  }

  test("select") {
    val testFrame = new BaseDFrame()

    // Mixing integral and named subscripts
    val selectedFrame = testFrame.select(
      "ACol", 5, "BCol", -"CCol", -4
    )

  }



  test("memory access speed") {

    import ExecutionContext.Implicits.global

    val rnd = new Random(1234L)

    val s = 1 << 30
    val blockSize = 1 << 6
    val numBlocks = s / blockSize
    val reads = 10 * (s / blockSize)

    val memchunk = ByteBuffer.allocate(s)

    // fill the chunk with random stuff
    while (memchunk.remaining() > 0) memchunk.put(rnd.nextInt().toByte)

    val arr = memchunk.array()

    var ms = System.currentTimeMillis()

    val futures = (0 until 4).map((s) => future {
      var sum = 0.0
      var i = 0
      var blockBase = 0
      while (i < reads) {
        var j = 0
        var l = 0l
        while (j < blockSize) {
          l = arr(blockBase + j)
          l << 3
          l |= arr(blockBase + j + 1) & 0xff
          l << 3
          l |= arr(blockBase + j + 2) & 0xff
          l << 3
          l |= arr(blockBase + j + 3) & 0xff
          l << 3
          l |= arr(blockBase + j + 4) & 0xff
          l << 3
          l |= arr(blockBase + j + 5) & 0xff
          l << 3
          l |= arr(blockBase + j + 6) & 0xff
          l << 3
          l |= arr(blockBase + j + 7) & 0xff

          sum += java.lang.Double.longBitsToDouble(l)
          j += 8
        }
        blockBase = blockSize * math.abs(((l ^ (l >> s + 1)).toInt) % numBlocks)
        i += 1
      }
    })
    futures.foreach(Await.result(_, atMost = Duration.Inf))
    ms = System.currentTimeMillis() - ms

    printf("N random %d ms.\n", ms)

  }

}


package com.example.snnser.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

/**
 * Runs the exported spiking network on-device.
 *
 * Drop the trained model at app/src/main/assets/snn_ser.onnx. Until then the
 * app should keep using [StubEmotionClassifier]; [isModelPresent] tells you
 * which one to build.
 */
class OnnxEmotionClassifier private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val expectedSamples: Int,
) : EmotionClassifier {

    override suspend fun classify(pcm: FloatArray): InferenceResult = withContext(Dispatchers.Default) {
        require(pcm.size == expectedSamples) {
            "Model expects $expectedSamples samples, got ${pcm.size}"
        }
        val started = System.nanoTime()

        val shape = longArrayOf(1, expectedSamples.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(pcm), shape).use { input ->
            session.run(mapOf(INPUT_PCM to input)).use { output ->
                @Suppress("UNCHECKED_CAST")
                val logits = (output[0].value as Array<FloatArray>)[0]

                val spikes = runCatching {
                    when (val v = output.get(OUTPUT_SPIKES).orElse(null)?.value) {
                        is LongArray -> v.firstOrNull()
                        is Array<*> -> (v.firstOrNull() as? Number)?.toLong()
                        is Number -> v.toLong()
                        else -> null
                    }
                }.getOrNull()

                InferenceResult(
                    ranked = rank(softmax(logits)),
                    latencyMs = (System.nanoTime() - started) / 1_000_000,
                    spikeCount = spikes,
                )
            }
        }
    }

    override fun close() {
        runCatching { session.close() }
    }

    companion object {
        const val MODEL_ASSET = "snn_ser.onnx"
        private const val INPUT_PCM = "pcm"
        private const val OUTPUT_SPIKES = "spikes"

        fun isModelPresent(context: Context): Boolean =
            runCatching { context.assets.list("")?.contains(MODEL_ASSET) == true }.getOrDefault(false)

        fun create(context: Context, expectedSamples: Int): OnnxEmotionClassifier {
            val bytes = context.assets.open(MODEL_ASSET).use { it.readBytes() }
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                // Small model, and thread churn costs more than it saves here.
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            return OnnxEmotionClassifier(env, env.createSession(bytes, options), expectedSamples)
        }
    }
}

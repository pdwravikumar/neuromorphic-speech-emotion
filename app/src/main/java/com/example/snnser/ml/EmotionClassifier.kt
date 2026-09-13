package com.example.snnser.ml

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.exp
import kotlin.random.Random

/** RAVDESS emotion set, in the label order the model was trained on. */
enum class Emotion(val label: String) {
    NEUTRAL("Neutral"),
    CALM("Calm"),
    HAPPY("Happy"),
    SAD("Sad"),
    ANGRY("Angry"),
    FEARFUL("Fearful"),
    DISGUST("Disgust"),
    SURPRISED("Surprised"),
}

data class Prediction(
    val emotion: Emotion,
    val confidence: Float,
)

data class InferenceResult(
    val ranked: List<Prediction>,
    val latencyMs: Long,
    /**
     * Total output spikes emitted during the forward pass. This is the number
     * that makes the neuromorphic claim measurable — log it, chart it, put it
     * in the README next to the CNN-LSTM MAC count.
     */
    val spikeCount: Long? = null,
) {
    val top: Prediction get() = ranked.first()
}

/**
 * Contract between the app and whatever is doing the classifying.
 *
 * The ONNX graph takes raw PCM, not features: mel-spectrogram extraction is
 * baked into the exported model, so there is no DSP on this side and no way
 * for training and inference features to drift apart.
 *
 *   input   "pcm"     float32 [1, 48000]   normalized to [-1, 1] @ 16 kHz
 *   output  "logits"  float32 [1, 8]       pre-softmax, Emotion ordinal order
 *   output  "spikes"  int64   [1]          optional, total output spikes
 */
interface EmotionClassifier : AutoCloseable {
    suspend fun classify(pcm: FloatArray): InferenceResult
}

internal fun softmax(logits: FloatArray): FloatArray {
    val max = logits.max()
    val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
    val sum = exps.sum()
    return FloatArray(exps.size) { exps[it] / sum }
}

internal fun rank(probs: FloatArray): List<Prediction> =
    Emotion.entries
        .mapIndexed { i, e -> Prediction(e, probs.getOrElse(i) { 0f }) }
        .sortedByDescending { it.confidence }

/**
 * Placeholder that lets the entire app be built, styled and demoed before the
 * SNN exists. Derives a stable fake answer from the audio's own energy so the
 * UI reacts to real input instead of flickering at random.
 *
 * Swap for [OnnxEmotionClassifier] once snn_ser.onnx lands in assets/.
 */
class StubEmotionClassifier(seed: Int = 7) : EmotionClassifier {
    private val random = Random(seed)

    override suspend fun classify(pcm: FloatArray): InferenceResult = withContext(Dispatchers.Default) {
        val started = System.currentTimeMillis()
        delay(350) // pretend inference takes a moment

        val energy = pcm.sumOf { abs(it).toDouble() } / pcm.size
        val logits = FloatArray(Emotion.entries.size) {
            (energy * 12.0).toFloat() * (it % 3) + random.nextFloat()
        }
        InferenceResult(
            ranked = rank(softmax(logits)),
            latencyMs = System.currentTimeMillis() - started,
            spikeCount = null,
        )
    }

    override fun close() = Unit
}

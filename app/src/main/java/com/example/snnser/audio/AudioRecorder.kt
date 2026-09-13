package com.example.snnser.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext
import kotlin.math.abs

/**
 * Captures a fixed-length mono window of speech and returns it as normalized
 * float PCM in [-1, 1].
 *
 * The length is fixed on purpose: the SNN is exported with a static number of
 * timesteps, so the model expects exactly [numSamples] floats every time.
 * Short utterances are zero-padded; long ones are cut off.
 */
class AudioRecorder(
    val sampleRate: Int = 16_000,
    val durationSec: Float = 3.0f,
) {
    val numSamples: Int = (sampleRate * durationSec).toInt()

    @SuppressLint("MissingPermission")
    suspend fun record(onLevel: (Float) -> Unit = {}): FloatArray = withContext(Dispatchers.IO) {
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        check(minBuf > 0) { "AudioRecord is unavailable on this device" }

        val chunk = 1024
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, chunk * 4) * 2,
        )
        check(recorder.state == AudioRecord.STATE_INITIALIZED) { "Could not initialise microphone" }

        val out = FloatArray(numSamples)
        val buf = ShortArray(chunk)
        var written = 0

        try {
            recorder.startRecording()
            while (written < numSamples) {
                coroutineContext.ensureActive()
                val read = recorder.read(buf, 0, chunk)
                if (read <= 0) break

                var peak = 0f
                for (i in 0 until read) {
                    if (written + i >= numSamples) break
                    val v = buf[i] / 32768.0f
                    out[written + i] = v
                    val a = abs(v)
                    if (a > peak) peak = a
                }
                written += read
                onLevel(peak)
            }
        } finally {
            runCatching { recorder.stop() }
            recorder.release()
        }

        // Peak-normalize so loud and quiet speakers land in a similar range.
        // Mirror this exactly in the training pipeline.
        val peak = out.maxOf { abs(it) }
        if (peak > 1e-5f) {
            for (i in out.indices) out[i] = out[i] / peak
        }
        out
    }
}

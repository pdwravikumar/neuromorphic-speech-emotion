package com.example.snnser.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.snnser.audio.AudioRecorder
import com.example.snnser.ml.EmotionClassifier
import com.example.snnser.ml.InferenceResult
import com.example.snnser.ml.OnnxEmotionClassifier
import com.example.snnser.ml.StubEmotionClassifier
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface UiState {
    data object Idle : UiState
    data class Recording(val level: Float, val progress: Float) : UiState
    data object Analyzing : UiState
    data class Done(val result: InferenceResult) : UiState
    data class Failed(val message: String) : UiState
}

class EmotionViewModel(app: Application) : AndroidViewModel(app) {

    private val recorder = AudioRecorder()

    /** Real model if it has been shipped, stub otherwise. */
    val usingRealModel: Boolean = OnnxEmotionClassifier.isModelPresent(app)

    private val classifier: EmotionClassifier =
        if (usingRealModel) {
            OnnxEmotionClassifier.create(app, recorder.numSamples)
        } else {
            StubEmotionClassifier()
        }

    private val _state = MutableStateFlow<UiState>(UiState.Idle)
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var job: Job? = null

    fun start() {
        if (job?.isActive == true) return
        job = viewModelScope.launch {
            _state.value = UiState.Recording(level = 0f, progress = 0f)
            try {
                var captured = 0
                val pcm = recorder.record { level ->
                    captured += 1024
                    _state.value = UiState.Recording(
                        level = level,
                        progress = (captured.toFloat() / recorder.numSamples).coerceIn(0f, 1f),
                    )
                }
                _state.value = UiState.Analyzing
                _state.value = UiState.Done(classifier.classify(pcm))
            } catch (t: Throwable) {
                _state.value = UiState.Failed(t.message ?: "Something went wrong")
            }
        }
    }

    fun reset() {
        job?.cancel()
        _state.value = UiState.Idle
    }

    override fun onCleared() {
        job?.cancel()
        classifier.close()
        super.onCleared()
    }
}

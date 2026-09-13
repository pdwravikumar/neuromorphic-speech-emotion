package com.example.snnser.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.snnser.ml.InferenceResult
import com.example.snnser.ml.Prediction

@Composable
fun RecordScreen(
    state: UiState,
    usingRealModel: Boolean,
    onRecord: () -> Unit,
    onReset: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Spacer(Modifier.height(8.dp))
        Text(
            text = "Neuromorphic SER",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = if (usingRealModel) "Spiking network · on-device" else "Stub model · no SNN loaded",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(Modifier.weight(1f))

        when (state) {
            is UiState.Done -> ResultBlock(state.result)
            is UiState.Failed -> Text(
                state.message,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
            )
            is UiState.Analyzing -> {
                CircularProgressIndicator()
                Spacer(Modifier.height(16.dp))
                Text("Running inference…", style = MaterialTheme.typography.bodyMedium)
            }
            else -> MicButton(state = state, onClick = onRecord)
        }

        Spacer(Modifier.weight(1f))

        if (state is UiState.Done || state is UiState.Failed) {
            Button(onClick = onReset, modifier = Modifier.fillMaxWidth()) {
                Text("Record again")
            }
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun MicButton(state: UiState, onClick: () -> Unit) {
    val recording = state as? UiState.Recording
    val scale by animateFloatAsState(
        targetValue = 1f + (recording?.level ?: 0f) * 0.35f,
        label = "mic-pulse",
    )

    Box(contentAlignment = Alignment.Center) {
        Box(
            modifier = Modifier
                .size(150.dp)
                .scale(if (recording != null) scale else 1f)
                .clip(CircleShape)
                .background(
                    if (recording != null) MaterialTheme.colorScheme.errorContainer
                    else MaterialTheme.colorScheme.primaryContainer
                )
                .clickable(enabled = recording == null, onClick = onClick),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = if (recording != null) "Listening" else "Tap to\nspeak",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Medium,
            )
        }
    }

    if (recording != null) {
        Spacer(Modifier.height(24.dp))
        LinearProgressIndicator(
            progress = { recording.progress },
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(RoundedCornerShape(3.dp)),
        )
    }
}

@Composable
private fun ResultBlock(result: InferenceResult) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = result.top.emotion.label,
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = "${(result.top.confidence * 100).toInt()}% confident",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(Modifier.height(28.dp))

        result.ranked.take(4).forEach { ConfidenceRow(it) }

        Spacer(Modifier.height(20.dp))
        Text(
            text = buildString {
                append("${result.latencyMs} ms")
                result.spikeCount?.let { append("  ·  $it spikes") }
            },
            style = MaterialTheme.typography.labelMedium,
            fontFamily = FontFamily.Monospace,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun ConfidenceRow(p: Prediction) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = p.emotion.label,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.width(88.dp),
        )
        LinearProgressIndicator(
            progress = { p.confidence },
            modifier = Modifier
                .weight(1f)
                .height(10.dp)
                .clip(RoundedCornerShape(5.dp)),
        )
        Text(
            text = "${(p.confidence * 100).toInt()}%",
            style = MaterialTheme.typography.labelSmall,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier
                .width(44.dp)
                .padding(start = 8.dp),
        )
    }
}

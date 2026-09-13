package com.example.snnser

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.snnser.ui.EmotionViewModel
import com.example.snnser.ui.RecordScreen

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    App()
                }
            }
        }
    }
}

@Composable
private fun App(vm: EmotionViewModel = viewModel()) {
    val context = LocalContext.current
    val state by vm.state.collectAsState()

    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    var denied by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { ok ->
        granted = ok
        denied = !ok
        if (ok) vm.start()
    }

    if (denied && !granted) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "Microphone access is needed to analyse speech. Nothing is recorded to disk or " +
                    "sent anywhere — inference runs entirely on this device.",
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(20.dp))
            Button(onClick = { launcher.launch(Manifest.permission.RECORD_AUDIO) }) {
                Text("Grant access")
            }
        }
        return
    }

    RecordScreen(
        state = state,
        usingRealModel = vm.usingRealModel,
        onRecord = {
            if (granted) vm.start() else launcher.launch(Manifest.permission.RECORD_AUDIO)
        },
        onReset = vm::reset,
    )
}

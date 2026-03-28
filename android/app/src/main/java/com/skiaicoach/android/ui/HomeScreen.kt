package com.skiaicoach.android.ui

import android.view.ViewGroup
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.media3.common.MediaItem
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView

@OptIn(ExperimentalMaterial3Api::class)
@androidx.annotation.OptIn(UnstableApi::class)
@Composable
fun HomeScreen(vm: MainViewModel) {
    val state by vm.state.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("SkiAI Coach") },
                actions = {
                    IconButton(onClick = { vm.refreshJobs() }) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh jobs")
                    }
                },
            )
        },
    ) { inner ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(inner),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                Text(
                    "Backend base URL (no trailing slash). Emulator: http://10.0.2.2:8001 — device on LAN: http://YOUR_PC_IP:8001",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            item {
                OutlinedTextField(
                    value = state.baseUrl,
                    onValueChange = vm::setBaseUrl,
                    label = { Text("Base URL") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                )
            }
            item {
                val pickVideo = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.PickVisualMedia(),
                ) { uri ->
                    if (uri != null) vm.uploadVideo(uri)
                }
                Button(
                    onClick = {
                        pickVideo.launch(
                            PickVisualMediaRequest(
                                ActivityResultContracts.PickVisualMedia.VideoOnly,
                            ),
                        )
                    },
                    enabled = !state.busy,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(Icons.Default.VideoLibrary, contentDescription = null)
                    Text(" Pick video & upload", modifier = Modifier.padding(start = 8.dp))
                }
            }
            item {
                if (state.busy) {
                    RowCenteredProgress()
                }
                Text(state.statusMessage, style = MaterialTheme.typography.bodyMedium)
                state.pickedLabel?.let { Text("File: $it", style = MaterialTheme.typography.labelMedium) }
            }
            if (state.recentJobs.isNotEmpty()) {
                item {
                    Text("Recent jobs", style = MaterialTheme.typography.titleMedium)
                }
                items(state.recentJobs, key = { it.job_id ?: "" }) { job ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable(enabled = !job.job_id.isNullOrBlank()) {
                                vm.selectJob(job.job_id!!)
                            },
                    ) {
                        Column(Modifier.padding(12.dp)) {
                            Text(job.filename ?: job.job_id ?: "—")
                            Text(
                                job.status ?: "",
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                    }
                }
            }
            state.loadJobsError?.let { err ->
                item { Text(err, color = MaterialTheme.colorScheme.error) }
            }
            state.completed?.summary?.let { s ->
                item {
                    Text("Metrics", style = MaterialTheme.typography.titleMedium)
                    Text("Carving score: ${s.carving_score?.toInt() ?: "—"}/100")
                    Text("Max edge: ${s.max_edge_inclination_deg ?: "—"}°")
                    Text("Backseat: ${s.backseat_percentage ?: "—"}%")
                }
            }
            if (state.feedbackText.isNotBlank()) {
                item {
                    Text("Coach feedback", style = MaterialTheme.typography.titleMedium)
                    Text(state.feedbackText)
                }
            }
            state.videoPlayUrl?.let { url ->
                item {
                    Text("Analyzed replay", style = MaterialTheme.typography.titleMedium)
                    VideoPlayer(url = url, modifier = Modifier.fillMaxWidth().height(220.dp))
                }
            }
        }
    }
}

@Composable
private fun RowCenteredProgress() {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator()
    }
}

@androidx.annotation.OptIn(UnstableApi::class)
@Composable
private fun VideoPlayer(url: String, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val exo = remember(url) {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(url))
            prepare()
        }
    }
    DisposableEffect(exo) {
        onDispose { exo.release() }
    }
    AndroidView(
        factory = { ctx ->
            PlayerView(ctx).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT,
                )
                player = exo
                useController = true
            }
        },
        modifier = modifier,
    )
}

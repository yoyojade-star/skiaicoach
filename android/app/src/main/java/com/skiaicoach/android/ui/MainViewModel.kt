package com.skiaicoach.android.ui

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.skiaicoach.android.BuildConfig
import com.skiaicoach.android.data.JobListItemDto
import com.skiaicoach.android.data.JobResultDto
import com.skiaicoach.android.data.createSkiApi
import com.skiaicoach.android.data.normalizeBaseUrl
import com.skiaicoach.android.data.toFeedbackDisplayText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

data class MainUiState(
    val baseUrl: String = BuildConfig.DEFAULT_BASE_URL.trim().removeSuffix("/"),
    val statusMessage: String = "Pick a video and upload (backend on port 8001).",
    val busy: Boolean = false,
    val jobId: String? = null,
    val pickedLabel: String? = null,
    val completed: JobResultDto? = null,
    val feedbackText: String = "",
    val videoPlayUrl: String? = null,
    val recentJobs: List<JobListItemDto> = emptyList(),
    val loadJobsError: String? = null,
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(MainUiState())
    val state: StateFlow<MainUiState> = _state.asStateFlow()

    init {
        refreshJobs()
    }

    fun setBaseUrl(url: String) {
        _state.update { it.copy(baseUrl = url.trim().removeSuffix("/")) }
    }

    fun refreshJobs() {
        val base = normalizeBaseUrlForApi()
        viewModelScope.launch {
            try {
                val api = createSkiApi(base)
                val jobs = api.listJobs()
                    .filter { !it.job_id.isNullOrBlank() }
                    .sortedByDescending { it.job_id }
                    .take(20)
                _state.update {
                    it.copy(recentJobs = jobs, loadJobsError = null)
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(loadJobsError = e.message ?: "Failed to load /jobs")
                }
            }
        }
    }

    fun selectJob(jobId: String) {
        _state.update {
            it.copy(
                jobId = jobId,
                statusMessage = "Loading job $jobId…",
                completed = null,
                feedbackText = "",
                videoPlayUrl = null,
            )
        }
        pollUntilTerminal(jobId)
    }

    fun uploadVideo(uri: Uri) {
        val app = getApplication<Application>()
        val base = normalizeBaseUrlForApi()
        val fileName = queryDisplayName(uri) ?: "upload.mp4"
        val ext = fileName.substringAfterLast('.', "mp4").lowercase()
        val mime = when (ext) {
            "mov" -> "video/quicktime"
            "webm" -> "video/webm"
            "mkv" -> "video/x-matroska"
            "avi" -> "video/x-msvideo"
            else -> "video/mp4"
        }

        viewModelScope.launch {
            _state.update {
                it.copy(
                    busy = true,
                    statusMessage = "Uploading…",
                    pickedLabel = fileName,
                    completed = null,
                    feedbackText = "",
                    videoPlayUrl = null,
                )
            }
            try {
                val bytes = withContext(Dispatchers.IO) {
                    app.contentResolver.openInputStream(uri)?.use { it.readBytes() }
                        ?: throw IOException("Cannot read video")
                }
                val body = bytes.toRequestBody(mime.toMediaTypeOrNull())
                val part = MultipartBody.Part.createFormData("file", fileName, body)
                val api = createSkiApi(base)
                val resp = withContext(Dispatchers.IO) { api.uploadVideo(part) }
                val jid = resp.job_id
                if (jid.isNullOrBlank()) {
                    _state.update {
                        it.copy(busy = false, statusMessage = "Upload OK but no job_id in response.")
                    }
                    return@launch
                }
                _state.update {
                    it.copy(
                        jobId = jid,
                        statusMessage = "Processing…",
                    )
                }
                pollUntilTerminal(jid)
            } catch (e: Exception) {
                _state.update {
                    it.copy(
                        busy = false,
                        statusMessage = "Upload failed: ${e.message}",
                    )
                }
            }
        }
    }

    private fun normalizeBaseUrlForApi(): String {
        val raw = _state.value.baseUrl.trim().ifBlank { "http://10.0.2.2:8001" }
        return normalizeBaseUrl(raw)
    }

    private fun pollUntilTerminal(jobId: String) {
        val base = normalizeBaseUrlForApi()
        viewModelScope.launch {
            _state.update { it.copy(busy = true) }
            try {
                val api = createSkiApi(base)
                while (isActive) {
                    val result = withContext(Dispatchers.IO) { api.getResult(jobId) }
                    when (result.status) {
                        "completed" -> {
                            val feedback = result.feedback.toFeedbackDisplayText()
                            val videoUrl = "${base.removeSuffix("/")}/uploads/${jobId}_analyzed.mp4"
                            _state.update {
                                it.copy(
                                    busy = false,
                                    statusMessage = "Analysis complete.",
                                    completed = result,
                                    feedbackText = feedback,
                                    videoPlayUrl = videoUrl,
                                )
                            }
                            refreshJobs()
                            return@launch
                        }
                        "failed" -> {
                            _state.update {
                                it.copy(
                                    busy = false,
                                    statusMessage = "Analysis failed: ${result.error ?: "unknown"}",
                                    completed = result,
                                )
                            }
                            return@launch
                        }
                        "not_found" -> {
                            _state.update {
                                it.copy(
                                    busy = false,
                                    statusMessage = "Job not found.",
                                )
                            }
                            return@launch
                        }
                        else -> {
                            _state.update {
                                it.copy(statusMessage = "Processing… (${result.status ?: "…"})")
                            }
                            delay(3000L)
                        }
                    }
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(
                        busy = false,
                        statusMessage = "Poll error: ${e.message}",
                    )
                }
            }
        }
    }

    private fun queryDisplayName(uri: Uri): String? {
        val app = getApplication<Application>()
        val cr = app.contentResolver
        cr.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && c.moveToFirst()) return c.getString(idx)
        }
        return uri.lastPathSegment
    }

    companion object {
        fun factory(app: Application): ViewModelProvider.Factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return MainViewModel(app) as T
            }
        }
    }
}

package com.skiaicoach.android.data

import com.google.gson.JsonElement

data class UploadResponse(
    val job_id: String?,
    val status: String?,
)

data class SummaryDto(
    val carving_score: Double?,
    val max_edge_inclination_deg: Double?,
    val backseat_percentage: Double?,
    val breaking_at_waist_percentage: Double?,
    val average_active_edge_deg: Double?,
    val duration_frames: Double?,
)

data class JobResultDto(
    val status: String?,
    val job_id: String?,
    val summary: SummaryDto?,
    val feedback: JsonElement?,
    val error: String?,
    val filename: String?,
)

data class JobListItemDto(
    val job_id: String?,
    val status: String?,
    val filename: String?,
)

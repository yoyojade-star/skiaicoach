package com.skiaicoach.android.data

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import java.util.concurrent.TimeUnit

interface SkiApiService {
    @Multipart
    @POST("upload")
    suspend fun uploadVideo(@Part file: MultipartBody.Part): UploadResponse

    @GET("result/{jobId}")
    suspend fun getResult(@Path("jobId") jobId: String): JobResultDto

    @GET("jobs")
    suspend fun listJobs(): List<JobListItemDto>
}

fun normalizeBaseUrl(url: String): String {
    val t = url.trim()
    return if (t.endsWith("/")) t else "$t/"
}

fun createSkiApi(baseUrl: String): SkiApiService {
    val logging = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BASIC
    }
    val client = OkHttpClient.Builder()
        .addInterceptor(logging)
        .connectTimeout(120, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .build()

    return Retrofit.Builder()
        .baseUrl(normalizeBaseUrl(baseUrl))
        .client(client)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(SkiApiService::class.java)
}

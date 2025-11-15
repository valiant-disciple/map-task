package com.example.wearhr

import android.util.Log
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class HttpClient {
    private val client = OkHttpClient()
    private val jsonMedia = "application/json; charset=utf-8".toMediaType()

    fun postHr(baseUrl: String, payload: HrPayload) {
        try {
            val body = JSONObject()
                .put("deviceId", payload.deviceId)
                .put("ts", payload.ts)
                .put("hr", payload.hr)
                .put("accuracy", payload.accuracy)
                .toString()
                .toRequestBody(jsonMedia)

            val request = Request.Builder()
                .url(baseUrl.trimEnd('/') + "/api/hr")
                .post(body)
                .build()

            client.newCall(request).enqueue(object : okhttp3.Callback {
                override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                    Log.w(TAG, "postHr failed: ${'$'}e")
                }

                override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                    response.use { Log.d(TAG, "postHr code: ${'$'}{response.code}") }
                }
            })
        } catch (e: Exception) {
            Log.w(TAG, "postHr exception: ${'$'}e")
        }
    }

    companion object {
        private const val TAG = "HttpClient"
    }
}



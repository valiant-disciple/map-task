package com.example.wearhr

data class HrPayload(
    val deviceId: String,
    val ts: Long,
    val hr: Int,
    val accuracy: Int
)



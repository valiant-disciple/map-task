data class SensorPayload(
    val deviceId: String,
    val ts: Long,
    val type: String,
    val values: List<Float>,
    val accuracy: Int
)



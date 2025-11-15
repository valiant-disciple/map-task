package com.example.wearhr

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.SystemClock
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : Activity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var heartRateSensor: Sensor? = null

    private lateinit var txtStatus: TextView
    private lateinit var txtHr: TextView
    private lateinit var txtTs: TextView
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button

    private val http = HttpClient()
    private var tracking = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        txtStatus = findViewById(R.id.txtStatus)
        txtHr = findViewById(R.id.txtHr)
        txtTs = findViewById(R.id.txtTs)
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)

        btnStart.setOnClickListener { startTracking() }
        btnStop.setOnClickListener { stopTracking() }

        ensurePermissions()
    }

    private fun ensurePermissions() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.BODY_SENSORS), REQ_SENSORS)
        }
    }

    private fun startTracking() {
        if (tracking) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) != PackageManager.PERMISSION_GRANTED) {
            txtStatus.text = "Status: Grant BODY_SENSORS"
            ensurePermissions()
            return
        }
        val sensor = heartRateSensor
        if (sensor == null) {
            txtStatus.text = "Status: HR sensor not available"
            return
        }
        tracking = true
        txtStatus.text = "Status: Tracking"
        sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)
    }

    private fun stopTracking() {
        if (!tracking) return
        tracking = false
        txtStatus.text = "Status: Stopped"
        sensorManager.unregisterListener(this)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_SENSORS) {
            val granted = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
            txtStatus.text = if (granted) "Status: Permission granted" else "Status: Permission denied"
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // no-op
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || event.sensor.type != Sensor.TYPE_HEART_RATE) return
        val hrValue = event.values.firstOrNull()?.toInt() ?: return

        // Convert sensor timestamp (ns since boot) to wall-clock ms
        val eventElapsedMs = event.timestamp / 1_000_000L
        val nowElapsedMs = SystemClock.elapsedRealtimeNanos() / 1_000_000L
        val wallTimeMs = System.currentTimeMillis() - (nowElapsedMs - eventElapsedMs)

        txtHr.text = "HR: ${'$'}hrValue"
        txtTs.text = "TS: ${'$'}wallTimeMs"

        val payload = HrPayload(
            deviceId = android.os.Build.MODEL ?: "watch",
            ts = wallTimeMs,
            hr = hrValue,
            accuracy = event.accuracy
        )
        http.postHr(BuildConfig.BASE_URL, payload)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopTracking()
    }

    companion object {
        private const val REQ_SENSORS = 1001
    }
}



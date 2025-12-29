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
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject

class MainActivity : Activity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var heartRateSensor: Sensor? = null
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null

    private lateinit var txtStatus: TextView
    private lateinit var txtHr: TextView
    private lateinit var txtTs: TextView
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button

    private val socketClient = SocketClient()
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
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        btnStart.setOnClickListener { startTracking() }
        btnStop.setOnClickListener { stopTracking() }

        // Initialize Socket
        socketClient.connect(BuildConfig.BASE_URL)
        socketClient.setOnCommandListener { command ->
            runOnUiThread {
                when (command) {
                    "start" -> startTracking()
                    "stop" -> stopTracking()
                }
            }
        }

        ensurePermissions()
        // Auto-start tracking when app opens
        startTracking()
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
        
        tracking = true
        txtStatus.text = "Status: Tracking"
        
        heartRateSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
        accelSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
        gyroSensor?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
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
        if (event == null) return
        
        val now = System.currentTimeMillis()
        val values = event.values.toList()
        
        // Update UI for HR only to keep it simple, or last sensor
        if (event.sensor.type == Sensor.TYPE_HEART_RATE) {
            txtHr.text = "HR: ${values[0].toInt()}"
        }
        txtTs.text = "TS: $now"

        val type = when(event.sensor.type) {
            Sensor.TYPE_HEART_RATE -> "HEART_RATE"
            Sensor.TYPE_ACCELEROMETER -> "ACCELEROMETER"
            Sensor.TYPE_GYROSCOPE -> "GYROSCOPE"
            else -> "UNKNOWN"
        }

        val payload = JSONObject().apply {
            put("deviceId", android.os.Build.MODEL ?: "watch")
            put("ts", now)
            put("type", type)
            put("values", JSONArray(values))
            put("accuracy", event.accuracy)
        }
        
        socketClient.send(payload.toString())
    }

    override fun onDestroy() {
        super.onDestroy()
        stopTracking()
        socketClient.close()
    }

    companion object {
        private const val REQ_SENSORS = 1001
    }
}

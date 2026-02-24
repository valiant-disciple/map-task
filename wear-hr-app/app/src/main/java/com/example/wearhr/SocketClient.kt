package com.example.wearhr

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okhttp3.Response
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import android.os.Handler
import android.os.Looper

class SocketClient {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    
    private var webSocket: WebSocket? = null
    private var commandListener: ((String) -> Unit)? = null
    private var isConnected = false
    private val reconnectHandler = Handler(Looper.getMainLooper())
    private var url: String = ""

    fun connect(baseUrl: String) {
        url = baseUrl
        val request = Request.Builder().url(baseUrl).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "Connected to $baseUrl")
                isConnected = true
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "Received: $text")
                try {
                    val json = JSONObject(text)
                    if (json.has("type") && json.getString("type") == "command") {
                        val action = json.getString("action")
                        // Post to main thread
                        Handler(Looper.getMainLooper()).post {
                            commandListener?.invoke(action)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing message", e)
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "Closing: $reason")
                isConnected = false
                // Reconnect after server-initiated close (e.g. Render redeployment)
                reconnectHandler.postDelayed({ connect(url) }, 3000)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "Failure: ${t.message}")
                isConnected = false
                // Try reconnect in 3s
                reconnectHandler.postDelayed({ connect(url) }, 3000)
            }
        })
    }

    fun send(data: Any) {
        if (!isConnected) return
        try {
            // If data is already a string, send it. If it's an object, stringify it.
            // For this app, we'll assume we construct the JSON string in MainActivity or here.
            // Let's expect a JSON string or stringifiable object.
            val message = if (data is String) data else data.toString()
            webSocket?.send(message)
        } catch (e: Exception) {
            Log.e(TAG, "Send failed", e)
        }
    }

    fun setOnCommandListener(listener: (String) -> Unit) {
        commandListener = listener
    }

    fun close() {
        webSocket?.close(1000, "App closed")
        reconnectHandler.removeCallbacksAndMessages(null)
    }

    companion object {
        private const val TAG = "SocketClient"
    }
}

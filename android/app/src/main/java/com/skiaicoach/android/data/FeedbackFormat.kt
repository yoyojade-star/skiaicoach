package com.skiaicoach.android.data

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive

fun JsonElement?.toFeedbackDisplayText(): String {
    if (this == null || isJsonNull) return ""
    if (this is JsonPrimitive && isString) return asString
    if (this is JsonObject) {
        val lines = mutableListOf<String>()
        for ((k, v) in entrySet()) {
            when {
                v.isJsonPrimitive && v.asJsonPrimitive.isString ->
                    lines.add("$k: ${v.asString}")
                v.isJsonArray -> lines.add("$k: ${v.asJsonArray}")
                else -> lines.add("$k: $v")
            }
        }
        if (lines.isNotEmpty()) return lines.joinToString("\n")
    }
    return toString()
}

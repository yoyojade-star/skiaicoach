package com.skiaicoach.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.skiaicoach.android.ui.HomeScreen
import com.skiaicoach.android.ui.MainViewModel
import com.skiaicoach.android.ui.theme.SkiCoachTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SkiCoachTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    val vm: MainViewModel = viewModel(factory = MainViewModel.factory(application))
                    HomeScreen(vm)
                }
            }
        }
    }
}

plugins {
    id("com.android.application")
}

// Номер сборки приходит из GitHub Actions (-PbuildNumber=...), он растёт с каждым запуском.
// Android ставит обновление поверх только если versionCode больше, а подпись та же.
val buildNumber = (project.findProperty("buildNumber") as String?)?.toIntOrNull() ?: 1

android {
    namespace = "app.devai"
    compileSdk = 34

    defaultConfig {
        applicationId = "app.devai"
        minSdk = 26
        targetSdk = 34
        versionCode = buildNumber
        versionName = "0.1-alpha.$buildNumber"
    }

    // Постоянный ключ подписи: благодаря ему каждая новая сборка ставится поверх старой.
    signingConfigs {
        create("devai") {
            storeFile = file("devai.keystore")
            storeType = "pkcs12"
            storePassword = "devai-sign"
            keyAlias = "devai"
            keyPassword = "devai-sign"
        }
    }

    buildTypes {
        getByName("debug") {
            signingConfig = signingConfigs.getByName("devai")
        }
    }

    // Движок модели лежит в jniLibs под видом библиотеки. Чтобы Android мог его запустить,
    // файлы должны быть распакованы на диск, а не читаться прямо из apk.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
